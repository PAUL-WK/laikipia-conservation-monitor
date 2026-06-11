"""
Laikipia Conservation Monitor
============================================
Streamlit application that streams live Google Earth Engine imagery onto
an interactive map with full date-range comparison support.

Layers available
----------------
  1. NDVI              — MODIS MOD13A1  (16-day composite, 500 m)
  2. Rainfall          — CHIRPS Daily   (cumulative sum, ~5.5 km)
  3. Land Surface Temp — MODIS MOD11A2  (8-day, 1 km, Kelvin → °C)
  4. Elevation / DEM   — SRTM 30 m      (static, no date filter)
  5. EVI               — MODIS MOD13A1  (Enhanced Vegetation Index)
  6. Burned Area       — MODIS MCD64A1  (monthly, 500 m)
  7. True Colour       — Landsat 8/9 C2 (median, cloud-filtered)

Date Comparison
---------------
  • Single mode  : one date range → one map
  • Compare mode : Period A vs Period B → two maps side-by-side with
                   identical layer stack and a difference chart below

Run
---
    conservation_env\\Scripts\\streamlit.exe run gee_map_viewer\\app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import ee
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import math
from folium.plugins import MeasureControl, MiniMap, MousePosition, TimestampedGeoJson
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laikipia Conservation Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Sidebar ── */
  section[data-testid="stSidebar"]       { background:#0d1b2a; }
  section[data-testid="stSidebar"] *     { color:#e0e0e0 !important; }
  section[data-testid="stSidebar"] hr    { border-color:#2a4a6b; }

  /* ── Period headers ── */
  .period-header {
    font-size:.95rem; font-weight:700; padding:5px 12px;
    border-radius:6px; margin-bottom:6px; text-align:center;
  }
  .period-a { background:#1565c0; color:white !important; }
  .period-b { background:#b71c1c; color:white !important; }

  /* ── Left info drawer ── */
  .left-drawer {
    background:#ffffff; border-radius:12px;
    border:1px solid #e0e0e0;
    box-shadow:3px 3px 12px rgba(0,0,0,.12);
    padding:0; overflow:hidden; height:100%;
  }
  .drawer-header {
    background:#0d1b2a; color:white;
    padding:10px 14px; font-weight:700; font-size:.9rem;
    display:flex; justify-content:space-between; align-items:center;
  }
  .drawer-body  { padding:12px 14px; }
  .drawer-metric {
    background:#f8f9fa; border-radius:8px; padding:8px 12px;
    margin-bottom:6px; border-left:3px solid #1565c0;
  }
  .drawer-metric .label  { font-size:.72rem; color:#666; font-weight:600;
                           text-transform:uppercase; letter-spacing:.5px; }
  .drawer-metric .value  { font-size:1.05rem; font-weight:700; color:#0d1b2a; }
  .drawer-metric .delta  { font-size:.75rem; color:#2e7d32; margin-top:1px; }
  .drawer-section        { font-size:.75rem; font-weight:700; color:#555;
                           text-transform:uppercase; letter-spacing:.8px;
                           margin:10px 0 4px; }
  .tag-pill {
    display:inline-block; padding:2px 9px; border-radius:10px;
    font-size:.72rem; font-weight:600; margin:2px;
  }

  /* ── Bottom chart drawer ── */
  div[data-testid="stExpander"] > div:first-child {
    background:#0d1b2a !important; border-radius:10px 10px 0 0;
    padding:10px 16px !important;
  }
  div[data-testid="stExpander"] > div:first-child p,
  div[data-testid="stExpander"] > div:first-child svg { color:white !important; }
  div[data-testid="stExpander"] > div:last-child {
    border:1px solid #e0e0e0; border-top:none;
    border-radius:0 0 10px 10px; padding:16px !important;
  }

  /* ── Legend swatch ── */
  .legend-wrap {
    background:#f8f9fa; padding:8px 12px; border-radius:8px;
    font-size:11px; font-family:sans-serif; margin-bottom:6px;
    border:1px solid #e0e0e0;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
KEY_PATH   = ROOT / "covariate_extractor" / "secrets" / "gee_service_account.json"
SAMPLE_CSV = ROOT / "covariate_extractor" / "sample_fixes.csv"
PROJECT_ID = "ee-kimanipaul21"

LAIKIPIA_CENTER = [0.35, 37.05]
LAIKIPIA_ZOOM   = 9

ANIMAL_COLOURS = [
    "#e63946","#457b9d","#2a9d8f","#e9c46a",
    "#f4a261","#6d6875","#a8dadc","#ff6b6b",
]

# ══════════════════════════════════════════════════════════════════════════════
# GEE AUTH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to satellite data...")
def init_gee(key_path: Path, project: str) -> bool:
    try:
        import base64
        from google.oauth2 import service_account
        SCOPES = ["https://www.googleapis.com/auth/earthengine"]

        # Local dev: key file on disk
        if key_path.exists():
            info = json.loads(key_path.read_text(encoding="utf-8"))
        # Streamlit Cloud: base64-encoded JSON (avoids all TOML newline issues)
        elif "gee_key_b64" in st.secrets:
            info = json.loads(base64.b64decode(st.secrets["gee_key_b64"]).decode("utf-8"))
        # Streamlit Cloud: individual fields in [gee] section (fallback)
        elif "gee" in st.secrets:
            info = dict(st.secrets["gee"])
            if "private_key" in info:
                info["private_key"] = info["private_key"].replace("\\n", "\n")
        else:
            st.error("No satellite credentials found. Add gee_key_b64 to Streamlit secrets.")
            return False

        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        ee.Initialize(credentials=creds, project=project)
        ee.Number(1).getInfo()   # ping
        return True
    except Exception as exc:
        st.error(f"Satellite auth failed: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# LAYER CATALOGUE
# Each entry: id, label, factory_fn, sample_fn, unit, description
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _ndvi_tile(start: str, end: str) -> dict:
    img = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(start, end).select("NDVI")
        .map(lambda i: i.multiply(0.0001))
        .median().rename("NDVI")
    )
    vis = {"min": -0.1, "max": 0.9,
           "palette": ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "MODIS MOD13A1 · NASA",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=1800, show_spinner=False)
def _evi_tile(start: str, end: str) -> dict:
    img = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(start, end).select("EVI")
        .map(lambda i: i.multiply(0.0001))
        .median().rename("EVI")
    )
    vis = {"min": -0.1, "max": 0.8,
           "palette": ["#ffffe5","#f7fcb9","#addd8e","#41ab5d","#006837","#004529"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "MODIS MOD13A1 EVI · NASA",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=1800, show_spinner=False)
def _rainfall_tile(start: str, end: str) -> dict:
    img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end).select("precipitation")
        .sum().rename("rainfall_mm")
    )
    vis = {"min": 0, "max": 300,
           "palette": ["#f7fbff","#c6dbef","#6baed6","#2171b5","#084594","#08306b"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "CHIRPS Daily · UCSB",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=1800, show_spinner=False)
def _lst_tile(start: str, end: str) -> dict:
    img = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end).select("LST_Day_1km")
        .map(lambda i: i.multiply(0.02).subtract(273.15).rename("LST_C"))
        .mean()
    )
    vis = {"min": 15, "max": 50,
           "palette": ["#313695","#74add1","#ffffbf","#f46d43","#a50026"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "MODIS MOD11A2 LST · NASA",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=86400, show_spinner=False)
def _dem_tile(_start: str, _end: str) -> dict:
    """SRTM 30 m DEM — static, date params ignored."""
    img = ee.Image("USGS/SRTMGL1_003").select("elevation")
    vis = {"min": 500, "max": 3500,
           "palette": ["#006837","#1a9850","#66bd63","#ffffbf",
                       "#d9ef8b","#a6d96a","#fdae61","#d73027","#ffffff"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "SRTM 30m · USGS",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=1800, show_spinner=False)
def _burned_tile(start: str, end: str) -> dict:
    img = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(start, end).select("BurnDate")
        .max().rename("BurnDate")
    )
    vis = {"min": 1, "max": 366,
           "palette": ["#ffffcc","#ffeda0","#fed976","#fd8d3c","#e31a1c","#800026"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "MODIS MCD64A1 Burned Area · NASA",
            "palette": vis["palette"], "vmin": vis["min"], "vmax": vis["max"]}


@st.cache_data(ttl=1800, show_spinner=False)
def _landsat_tile(start: str, end: str) -> dict:
    def _prep(col_id: str) -> ee.ImageCollection:
        return (
            ee.ImageCollection(col_id)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .select(["SR_B4","SR_B3","SR_B2"])
            .map(lambda i: i.multiply(0.0000275).add(-0.2))
        )
    col = _prep("LANDSAT/LC09/C02/T1_L2")
    if col.size().getInfo() == 0:
        col = _prep("LANDSAT/LC08/C02/T1_L2")
    img = col.median()
    vis = {"min": 0.0, "max": 0.3, "bands": ["SR_B4","SR_B3","SR_B2"]}
    mid = img.getMapId(vis)
    return {"url": mid["tile_fetcher"].url_format,
            "attr": "Landsat 8/9 C2 · USGS",
            "palette": None, "vmin": None, "vmax": None}


# Catalogue: key → (display label, tile_factory, band_for_sampling, scale_m, unit)
LAYER_CATALOGUE: dict[str, tuple] = {
    "NDVI":         ("NDVI (MODIS)",              _ndvi_tile,    "NDVI",          500,  ""),
    "EVI":          ("EVI (MODIS)",               _evi_tile,     "EVI",           500,  ""),
    "Rainfall":     ("Rainfall mm (CHIRPS)",       _rainfall_tile,"precipitation", 5566, "mm"),
    "LST":          ("Land Surface Temp (MODIS)",  _lst_tile,     "LST_C",         1000, "°C"),
    "DEM":          ("Elevation / DEM (SRTM)",     _dem_tile,     "elevation",     30,   "m"),
    "Burned Area":  ("Burned Area (MODIS)",        _burned_tile,  "BurnDate",      500,  "day"),
    "True Colour":  ("True Colour (Landsat 8/9)",  _landsat_tile, None,            30,   ""),
}

LAYER_LABELS = {k: v[0] for k, v in LAYER_CATALOGUE.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES SAMPLERS (for the "between dates" chart)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def sample_ndvi_series(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Sample MODIS NDVI for every 16-day image in the date range.

    Root-cause fix: scale transforms (multiply) strip system:time_start from
    mapped images. We keep the raw collection intact, sample the raw DN value,
    and apply the 0.0001 scale factor in Python after the getInfo() call.
    """
    point = ee.Geometry.Point([lon, lat])
    col   = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(start, end)
        .select("NDVI")
        # NO .map() transform — preserves system:time_start on every image
    )

    def extract(img):
        v = img.sample(region=point, scale=500, numPixels=1).first().get("NDVI")
        return ee.Feature(None, {
            "date":  img.date().format("YYYY-MM-dd"),   # safe: no transform applied
            "value": v,
        })

    feats = col.map(extract).getInfo()["features"]
    rows  = []
    for f in feats:
        raw = f["properties"]["value"]
        if raw is not None:
            rows.append({
                "date": f["properties"]["date"],
                "NDVI": round(float(raw) * 0.0001, 4),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "NDVI"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def sample_rainfall_series(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Sample CHIRPS daily precipitation.  No scale transform needed (values are
    already in mm), but we still keep the collection untransformed so that
    system:time_start is always present.
    """
    point = ee.Geometry.Point([lon, lat])
    col   = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .select("precipitation")
    )

    def extract(img):
        v = img.sample(region=point, scale=5566, numPixels=1).first().get("precipitation")
        return ee.Feature(None, {
            "date":  img.date().format("YYYY-MM-dd"),
            "value": v,
        })

    feats = col.map(extract).getInfo()["features"]
    rows  = [
        {"date": f["properties"]["date"],
         "Rainfall_mm": float(f["properties"]["value"])}
        for f in feats if f["properties"]["value"] is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["date", "Rainfall_mm"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def sample_lst_series(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Sample MODIS MOD11A2 Land Surface Temperature (8-day composite).

    Raw DN → Kelvin: multiply 0.02
    Kelvin → Celsius: subtract 273.15
    Both conversions applied in Python after sampling raw DN to preserve
    system:time_start on every image.
    """
    point = ee.Geometry.Point([lon, lat])
    col   = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .select("LST_Day_1km")
        # NO .map() transform — preserves system:time_start
    )

    def extract(img):
        v = img.sample(region=point, scale=1000, numPixels=1).first().get("LST_Day_1km")
        return ee.Feature(None, {
            "date":  img.date().format("YYYY-MM-dd"),
            "value": v,
        })

    feats = col.map(extract).getInfo()["features"]
    rows  = []
    for f in feats:
        raw = f["properties"]["value"]
        if raw is not None:
            celsius = round(float(raw) * 0.02 - 273.15, 2)  # DN → °C in Python
            rows.append({"date": f["properties"]["date"], "LST_C": celsius})
    if not rows:
        return pd.DataFrame(columns=["date", "LST_C"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# POINT SAMPLER (single value for info panel)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def sample_point(layer_key: str, lat: float, lon: float,
                 start: str, end: str) -> float | None:
    _, factory, band, scale, _ = LAYER_CATALOGUE[layer_key]
    if band is None:
        return None
    try:
        point    = ee.Geometry.Point([lon, lat])
        tile_cfg = factory(start, end)   # gets cached image

        # Re-build the image directly for sampling
        if layer_key == "NDVI":
            img = (ee.ImageCollection("MODIS/061/MOD13A1")
                   .filterDate(start, end).select("NDVI")
                   .map(lambda i: i.multiply(0.0001)).median())
        elif layer_key == "EVI":
            img = (ee.ImageCollection("MODIS/061/MOD13A1")
                   .filterDate(start, end).select("EVI")
                   .map(lambda i: i.multiply(0.0001)).median())
        elif layer_key == "Rainfall":
            img = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                   .filterDate(start, end).select("precipitation").sum())
        elif layer_key == "LST":
            img = (ee.ImageCollection("MODIS/061/MOD11A2")
                   .filterDate(start, end).select("LST_Day_1km")
                   .map(lambda i: i.multiply(0.02).subtract(273.15)).mean()
                   .rename("LST_C"))
            band = "LST_C"
        elif layer_key == "DEM":
            img  = ee.Image("USGS/SRTMGL1_003").select("elevation")
        elif layer_key == "Burned Area":
            img = (ee.ImageCollection("MODIS/061/MCD64A1")
                   .filterDate(start, end).select("BurnDate").max())
        else:
            return None

        val = (img.sample(region=point, scale=scale, numPixels=1)
               .first().get(band).getInfo())
        return float(val) if val is not None else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GPS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_gps(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = df.columns.str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return df.sort_values(["animal_id","timestamp"]).reset_index(drop=True)


def add_tracks(m: folium.Map, df: pd.DataFrame,
               show_lines: bool, show_points: bool,
               selected: list[str]) -> None:
    animals    = df["animal_id"].unique().tolist()
    colour_map = {a: ANIMAL_COLOURS[i % len(ANIMAL_COLOURS)]
                  for i, a in enumerate(animals)}
    for animal in selected:
        sub    = df[df["animal_id"] == animal]
        colour = colour_map[animal]
        coords = list(zip(sub["latitude"], sub["longitude"]))
        if show_lines and len(coords) > 1:
            folium.PolyLine(coords, color=colour, weight=2.5,
                            opacity=0.85, tooltip=animal).add_to(m)
        if show_points:
            for _, row in sub.iterrows():
                ts = str(row["timestamp"])[:16]
                folium.CircleMarker(
                    [row["latitude"], row["longitude"]],
                    radius=4, color=colour, fill=True, fill_color=colour,
                    fill_opacity=0.9,
                    popup=folium.Popup(
                        f"<b>{animal}</b><br>{ts}<br>"
                        f"{row['latitude']:.4f}, {row['longitude']:.4f}",
                        max_width=160),
                    tooltip=f"{animal} {ts}",
                ).add_to(m)
        if coords:
            folium.Marker(coords[0],
                          icon=folium.Icon(color="green", icon="play", prefix="fa"),
                          tooltip=f"{animal} start").add_to(m)
            folium.Marker(coords[-1],
                          icon=folium.Icon(color="red",   icon="stop", prefix="fa"),
                          tooltip=f"{animal} last fix").add_to(m)


def add_animated_tracks(m: folium.Map, df: pd.DataFrame,
                        selected: list[str], speed: int) -> None:
    """Add TimestampedGeoJson time-slider animation to the map."""
    animals    = df["animal_id"].unique().tolist()
    colour_map = {a: ANIMAL_COLOURS[i % len(ANIMAL_COLOURS)]
                  for i, a in enumerate(animals)}

    filtered = df[df["animal_id"].isin(selected)].copy()

    # Dashed history lines (static, always visible)
    for animal in selected:
        sub    = filtered[filtered["animal_id"] == animal]
        coords = list(zip(sub["latitude"], sub["longitude"]))
        if len(coords) > 1:
            folium.PolyLine(
                coords, color=colour_map[animal], weight=2,
                opacity=0.4, dash_array="6 4",
                tooltip=f"{animal} full track",
            ).add_to(m)

    # Start marker per animal
    for animal in selected:
        sub = filtered[filtered["animal_id"] == animal]
        if sub.empty:
            continue
        first = sub.iloc[0]
        folium.CircleMarker(
            [first["latitude"], first["longitude"]],
            radius=6, color=colour_map[animal],
            fill=True, fill_color="#ffffff", fill_opacity=1, weight=2,
            tooltip=f"{animal} start · {str(first['timestamp'])[:10]}",
        ).add_to(m)

    # Animated markers
    features = []
    for _, row in filtered.iterrows():
        animal = row["animal_id"]
        ts     = row["timestamp"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "time":  ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "popup": (
                    f"<b>{animal}</b><br>"
                    f"{ts.strftime('%Y-%m-%d %H:%M')}<br>"
                    f"{row['latitude']:.4f}, {row['longitude']:.4f}"
                ),
                "icon": "circle",
                "iconstyle": {
                    "fillColor":   colour_map[animal],
                    "fillOpacity": 0.92,
                    "stroke":      True,
                    "color":       "#ffffff",
                    "weight":      2,
                    "radius":      10,
                },
            },
        })

    if features:
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT6H",
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=speed,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
            time_slider_drag_update=True,
            duration="P1D",
        ).add_to(m)


# ══════════════════════════════════════════════════════════════════════════════
# LEGEND HTML (injected into folium map)
# ══════════════════════════════════════════════════════════════════════════════

def _inject_dynamic_legend(m, active_layers, layer_cache):
    """Inject a legend into the folium map that shows immediately and
    updates when the user checks/unchecks layers in the in-map LayerControl.

    Layer order rule:
      - active_layers[0] = first in list = highest zIndex = visually on top
      - Legend always shows the topmost currently-checked layer.
    The legend div is rendered by Python (always visible on load).
    A small JS snippet listens to checkbox change events on the layer control
    to update the div without needing a Python round-trip.
    """
    ld = {}
    ordered_labels = []   # index 0 = topmost layer on map (highest zIndex)
    for key in active_layers:
        if key not in layer_cache or layer_cache[key]["palette"] is None:
            continue
        tile = layer_cache[key]
        _, _, _, _, unit = LAYER_CATALOGUE[key]
        lbl = LAYER_LABELS[key]
        ld[lbl] = {
            "label": lbl,
            "vmin":  tile["vmin"],
            "vmax":  tile["vmax"],
            "stops": ", ".join(tile["palette"]),
            "unit":  unit,
        }
        ordered_labels.append(lbl)

    if not ld:
        return

    # Initial legend content: first checked layer (index 0 = topmost)
    top    = ld[ordered_labels[0]]
    leg_id = f"legend_{m.get_name()}"

    def swatch(d):
        return (
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.6px;color:#444;margin-bottom:6px'>Legend</div>"
            "<b style='font-size:11px'>" + d["label"] + "</b>"
            "<div style='height:9px;width:140px;margin:3px 0;border-radius:3px;"
            "background:linear-gradient(to right," + d["stops"] + ")'></div>"
            "<div style='display:flex;justify-content:space-between;font-size:10px;color:#555'>"
            "<span>" + str(d["vmin"]) + " " + d["unit"] + "</span>"
            "<span>" + str(d["vmax"]) + " " + d["unit"] + "</span>"
            "</div>"
        )

    ld_js  = json.dumps(ld)
    ord_js = json.dumps(ordered_labels)

    html = f"""
<div id="{leg_id}" style="position:fixed;right:10px;bottom:30px;z-index:9999;
background:rgba(255,255,255,0.93);padding:10px 14px;border-radius:10px;
box-shadow:2px 2px 8px rgba(0,0,0,.25);font-family:sans-serif;
min-width:160px;max-width:190px">
{swatch(top)}
</div>
<script>
(function(){{
  var LD    = {ld_js};
  var ORDER = {ord_js};
  var el    = document.getElementById("{leg_id}");

  function topLayer() {{
    var labels = document.querySelectorAll(
      ".leaflet-control-layers-overlays label");
    var chk = {{}};
    labels.forEach(function(l) {{
      var cb = l.querySelector("input[type=\\"checkbox\\"]");
      chk[l.textContent.trim()] = cb && cb.checked;
    }});
    for (var i = 0; i < ORDER.length; i++) {{
      if (chk[ORDER[i]] !== false && LD[ORDER[i]]) return LD[ORDER[i]];
    }}
    return null;
  }}

  function render() {{
    var d = topLayer();
    if (!d) {{ el.style.display = "none"; return; }}
    el.style.display = "block";
    el.innerHTML =
      "<div style=\\"font-size:10px;font-weight:700;text-transform:uppercase;" +
      "letter-spacing:.6px;color:#444;margin-bottom:6px\\">Legend</div>" +
      "<b style=\\"font-size:11px\\">" + d.label + "</b>" +
      "<div style=\\"height:9px;width:140px;margin:3px 0;border-radius:3px;" +
      "background:linear-gradient(to right," + d.stops + ")\\"></div>" +
      "<div style=\\"display:flex;justify-content:space-between;" +
      "font-size:10px;color:#555\\">" +
      "<span>" + d.vmin + " " + d.unit + "</span>" +
      "<span>" + d.vmax + " " + d.unit + "</span></div>";
  }}

  function attach() {{
    var cbs = document.querySelectorAll(
      ".leaflet-control-layers-overlays input[type=\\"checkbox\\"]");
    if (cbs.length === 0) {{ setTimeout(attach, 300); return; }}
    cbs.forEach(function(cb) {{ cb.addEventListener("change", render); }});
    render();
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", attach);
  }} else {{
    setTimeout(attach, 300);
  }}
}})();
</script>"""
    m.get_root().html.add_child(folium.Element(html))


# ══════════════════════════════════════════════════════════════════════════════
# BUILD A SINGLE FOLIUM MAP
# ══════════════════════════════════════════════════════════════════════════════

def build_map(
    active_layers: list[str],
    start: str, end: str,
    opacity: float,
    basemap: str,
    gps_df,
    show_lines: bool, show_points: bool,
    selected_animals: list[str],
    height: int,
    live_tracking: bool = False,
    playback_speed: int = 5,
    label: str = "",
) -> tuple[folium.Map, dict[str, dict]]:
    """Build a folium map, return (map, layer_cache)."""

    basemap_tiles = {
        "CartoDB Positron":    "CartoDB positron",
        "CartoDB Dark Matter": "CartoDB dark_matter",
        "OpenStreetMap":       "OpenStreetMap",
        "ESRI Satellite":      (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri World Imagery"
        ),
    }

    tile_val = basemap_tiles[basemap]
    if isinstance(tile_val, tuple):
        m = folium.Map(location=LAIKIPIA_CENTER, zoom_start=LAIKIPIA_ZOOM,
                       tiles=None, control_scale=True)
        folium.TileLayer(tiles=tile_val[0], attr=tile_val[1],
                         name=basemap).add_to(m)
    else:
        m = folium.Map(location=LAIKIPIA_CENTER, zoom_start=LAIKIPIA_ZOOM,
                       tiles=tile_val, control_scale=True)

    layer_cache: dict[str, dict] = {}
    for key in active_layers:
        _, factory, _, _, _ = LAYER_CATALOGUE[key]
        try:
            info = factory(start, end)
            layer_cache[key] = info
            n_layers = len(active_layers)
            z_index  = 200 + (n_layers - 1 - list(active_layers).index(key))
            folium.TileLayer(
                tiles=info["url"],
                attr=info["attr"],
                name=LAYER_LABELS[key],
                overlay=True,
                opacity=opacity,
                zIndex=z_index,
            ).add_to(m)
        except Exception as exc:
            st.warning(f"Could not load {LAYER_LABELS[key]}: {exc}")

    if gps_df is not None and selected_animals:
        if live_tracking:
            add_animated_tracks(m, gps_df, selected_animals, playback_speed)
        else:
            add_tracks(m, gps_df, show_lines, show_points, selected_animals)

    # Dynamic legend — tracks the in-map LayerControl checkboxes via JS events
    _inject_dynamic_legend(m, active_layers, layer_cache)

    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MeasureControl(position="topleft").add_to(m)
    MousePosition(position="bottomright", prefix="Lat/Lon: ",
                  separator=" | ").add_to(m)

    # Period label overlay
    if label:
        colour = "#1565c0" if "A" in label else "#b71c1c"
        m.get_root().html.add_child(folium.Element(
            f"<div style='position:fixed;top:10px;left:60px;z-index:9999;"
            f"background:{colour};color:white;padding:4px 12px;"
            f"border-radius:4px;font-weight:700;font-size:13px;"
            f"font-family:sans-serif'>{label}</div>"
        ))

    return m, layer_cache


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "drawer_open":   False,   # left point-detail drawer
        "clicked_lat":   None,
        "clicked_lon":   None,
        "clicked_from":  None,    # "map" | "map_a" | "map_b"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# LEFT DRAWER — point detail panel
# ══════════════════════════════════════════════════════════════════════════════

def _metric_row(label: str, value: str, delta: str = "", delta_col: str = "") -> str:
    delta_html = (
        f"<div style='font-size:.72rem;color:{delta_col};margin-top:1px'>{delta}</div>"
        if delta else ""
    )
    return (
        f"<div class='drawer-metric'>"
        f"  <div style='font-size:.7rem;color:#666;font-weight:600;"
        f"       text-transform:uppercase;letter-spacing:.4px'>{label}</div>"
        f"  <div style='font-size:.95rem;font-weight:700;color:#0d1b2a'>{value}</div>"
        f"  {delta_html}"
        f"</div>"
    )


def _section(title: str) -> str:
    return (
        f"<div style='font-size:.68rem;font-weight:700;color:#555;"
        f"text-transform:uppercase;letter-spacing:.8px;margin:10px 0 4px'>"
        f"{title}</div>"
    )


def _render_left_drawer(
    gee_ok: bool,
    active_layers: list[str],
    layer_cache_a: dict, layer_cache_b: dict | None,
    gps_df,
    str_a: str, ste_a: str,
    str_b: str | None, ste_b: str | None,
    map_height: int = 600,
) -> None:
    lat = st.session_state.clicked_lat
    lon = st.session_state.clicked_lon

    # ── Collect all data BEFORE rendering ─────────────────────────────────────
    sampleable = [k for k in active_layers
                  if LAYER_CATALOGUE[k][2] is not None and k != "True Colour"]

    vals_a: dict[str, float | None] = {}
    vals_b: dict[str, float | None] = {}
    if gee_ok and sampleable:
        with st.spinner("Sampling satellite values…"):
            for key in sampleable:
                vals_a[key] = sample_point(key, lat, lon, str_a, ste_a)
                if str_b and ste_b:
                    vals_b[key] = sample_point(key, lat, lon, str_b, ste_b)

    nearest_html = ""
    if gps_df is not None:
        tmp = gps_df.copy()
        # Use proper haversine for nearest fix
        def _hav(r):
            R = 6371.0
            φ1, φ2 = math.radians(r["latitude"]), math.radians(lat)
            dφ = math.radians(lat - r["latitude"])
            dλ = math.radians(lon - r["longitude"])
            a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        tmp["_dist_km"] = tmp.apply(_hav, axis=1)
        nr      = tmp.nsmallest(1, "_dist_km").iloc[0]
        dist_km = round(nr["_dist_km"], 2)
        nearest_html = (
            _section("Nearest GPS Fix")
            + _metric_row("Animal",            nr["animal_id"])
            + _metric_row("Timestamp",         str(nr["timestamp"])[:16])
            + _metric_row("Distance from click", f"{dist_km} km")
        )

        # Movement stats from previous fix (useful in live tracking mode)
        animal_fixes = (
            gps_df[gps_df["animal_id"] == nr["animal_id"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        pos = animal_fixes.index[
            animal_fixes["timestamp"] == nr["timestamp"]
        ].tolist()
        if pos and pos[0] > 0:
            prev     = animal_fixes.iloc[pos[0] - 1]
            R        = 6371.0
            φ1, φ2  = math.radians(prev["latitude"]), math.radians(nr["latitude"])
            dφ       = math.radians(nr["latitude"]  - prev["latitude"])
            dλ       = math.radians(nr["longitude"] - prev["longitude"])
            a        = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
            step_km  = round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 3)
            hrs      = (nr["timestamp"] - prev["timestamp"]).total_seconds() / 3600
            kmh      = round(step_km / hrs, 2) if hrs > 0 else 0
            nearest_html += (
                _section("Movement from Previous Fix")
                + _metric_row("Step distance", f"{step_km} km")
                + _metric_row("Time gap",      f"{hrs:.1f} h")
                + _metric_row("Speed",         f"{kmh} km/h")
            )

    # ── Build HTML body ────────────────────────────────────────────────────────
    body = _section("Coordinates")
    body += _metric_row("Latitude",  f"{lat:.5f}°")
    body += _metric_row("Longitude", f"{lon:.5f}°")

    if gee_ok and sampleable:
        body += _section("Period A Values")
        for key in sampleable:
            _, _, _, _, unit = LAYER_CATALOGUE[key]
            label = LAYER_LABELS[key].split("(")[0].strip()
            v = vals_a.get(key)
            body += _metric_row(label, f"{v:.3f} {unit}" if v is not None else "No data")

        if str_b and ste_b:
            body += _section("Period B  ·  Change vs A")
            for key in sampleable:
                _, _, _, _, unit = LAYER_CATALOGUE[key]
                label = LAYER_LABELS[key].split("(")[0].strip()
                va, vb = vals_a.get(key), vals_b.get(key)
                if va is not None and vb is not None:
                    d     = vb - va
                    sign  = "+" if d >= 0 else ""
                    dcol  = "#2e7d32" if d >= 0 else "#c62828"
                    body += _metric_row(label, f"{vb:.3f} {unit}",
                                        delta=f"{sign}{d:.3f} vs A", delta_col=dcol)
                else:
                    body += _metric_row(label, "No data")

    body += nearest_html

    # ── Render as ONE html block with fixed height + internal scroll ───────────
    # Reserve ~42 px for the native close button that sits above this block.
    scroll_height = map_height - 42
    st.button(
        "✕  Close inspector", key="close_drawer", use_container_width=True,
        on_click=lambda: st.session_state.update(
            drawer_open=False, clicked_lat=None, clicked_lon=None
        ),
    )
    st.markdown(
        f"""
        <div style='height:{scroll_height}px;overflow-y:auto;
                    border:1px solid #e0e0e0;border-radius:0 0 12px 12px;
                    background:#fff;box-shadow:3px 3px 12px rgba(0,0,0,.1)'>
          <div style='background:#0d1b2a;color:white;padding:9px 14px;
                      font-weight:700;font-size:.88rem;letter-spacing:.3px'>
            📍 Point Inspector
          </div>
          <div style='padding:10px 14px'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM CHART DRAWER
# ══════════════════════════════════════════════════════════════════════════════

def _render_bottom_drawer(
    gee_ok: bool,
    gps_df,
    str_a: str, ste_a: str,
    str_b: str | None, ste_b: str | None,
) -> None:
    lat = st.session_state.clicked_lat
    lon = st.session_state.clicked_lon
    has_point = lat is not None and lon is not None

    drawer_label = (
        f"📊  Charts"
        + (f"  —  point ({lat:.3f}, {lon:.3f})" if has_point else
           "  —  click the map to enable time-series")
    )

    with st.expander(drawer_label, expanded=False):

        # ── Tab set ────────────────────────────────────────────────────────────
        tab_names = ["NDVI", "Rainfall", "Land Temp", "Compare", "GPS Analytics"]
        tabs      = st.tabs(tab_names)
        VAR_CFG   = {
            "NDVI":      ("NDVI",       "Rainfall_mm", "#2ca02c", "NDVI"),
            "Rainfall":  ("Rainfall",   "Rainfall_mm", "#1f77b4", "Rainfall (mm/day)"),
            "Land Temp": ("LST",        "LST_C",       "#d62728", "LST (°C)"),
        }

        for tab_name, tab in zip(tab_names, tabs):
            with tab:

                # ── Individual variable tabs ───────────────────────────────────
                if tab_name in VAR_CFG:
                    var_key, ycol, colour, ylab = VAR_CFG[tab_name]

                    if not gee_ok:
                        st.warning("Satellite data not connected.")
                        continue
                    if not has_point:
                        st.info("Click a point on the map above to see the time-series here.")
                        continue

                    with st.spinner(f"Fetching {tab_name} series..."):
                        if var_key == "NDVI":
                            df_a = sample_ndvi_series(lat, lon, str_a, ste_a)
                            df_b = sample_ndvi_series(lat, lon, str_b, ste_b) \
                                   if str_b else pd.DataFrame()
                        elif var_key == "Rainfall":
                            df_a = sample_rainfall_series(lat, lon, str_a, ste_a)
                            df_b = sample_rainfall_series(lat, lon, str_b, ste_b) \
                                   if str_b else pd.DataFrame()
                        else:
                            df_a = sample_lst_series(lat, lon, str_a, ste_a)
                            df_b = sample_lst_series(lat, lon, str_b, ste_b) \
                                   if str_b else pd.DataFrame()

                    # Normalise: ensure the value column exists even if df is non-empty
                    for _df in (df_a, df_b):
                        if not _df.empty and ycol not in _df.columns:
                            _df[ycol] = float("nan")

                    if df_a.empty and df_b.empty:
                        st.info(f"No {tab_name} data found for this point.")
                        continue

                    fig = go.Figure()
                    if not df_a.empty:
                        fig.add_trace(go.Scatter(
                            x=df_a["date"], y=df_a[ycol],
                            mode="lines+markers", name="Period A",
                            line=dict(color="#1565c0", width=2),
                            marker=dict(size=5),
                        ))
                    if not df_b.empty:
                        fig.add_trace(go.Scatter(
                            x=df_b["date"], y=df_b[ycol],
                            mode="lines+markers", name="Period B",
                            line=dict(color="#b71c1c", width=2, dash="dot"),
                            marker=dict(size=5, symbol="diamond"),
                        ))
                    fig.update_layout(
                        height=320,
                        xaxis_title="Date", yaxis_title=ylab,
                        margin=dict(t=20, b=20, l=50, r=20),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.08),
                        plot_bgcolor="#f8f9fa",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats row
                    stat_df = df_a if not df_a.empty else df_b
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Mean (A)",  f"{df_a[ycol].mean():.3f}" if not df_a.empty else "—")
                    s2.metric("Min  (A)",  f"{df_a[ycol].min():.3f}"  if not df_a.empty else "—")
                    s3.metric("Max  (A)",  f"{df_a[ycol].max():.3f}"  if not df_a.empty else "—")
                    s4.metric("Mean (B)",  f"{df_b[ycol].mean():.3f}" if not df_b.empty else "—")

                # ── Compare tab — all three variables together ─────────────────
                elif tab_name == "Compare":
                    if not gee_ok:
                        st.warning("Satellite data not connected.")
                        continue
                    if not has_point:
                        st.info("Click a point on the map to enable comparison.")
                        continue

                    with st.spinner("Fetching all series for comparison..."):
                        ndvi_a  = sample_ndvi_series(lat, lon, str_a, ste_a)
                        rain_a  = sample_rainfall_series(lat, lon, str_a, ste_a)
                        lst_a   = sample_lst_series(lat, lon, str_a, ste_a)
                        ndvi_b  = sample_ndvi_series(lat, lon, str_b, ste_b)  if str_b else pd.DataFrame()
                        rain_b  = sample_rainfall_series(lat, lon, str_b, ste_b) if str_b else pd.DataFrame()
                        lst_b   = sample_lst_series(lat, lon, str_b, ste_b)  if str_b else pd.DataFrame()

                    fig = go.Figure()

                    def _add(df, col, name, colour, dash="solid", symbol="circle"):
                        if not df.empty:
                            fig.add_trace(go.Scatter(
                                x=df["date"], y=df[col],
                                name=name, yaxis="y1",
                                mode="lines+markers",
                                line=dict(color=colour, width=2, dash=dash),
                                marker=dict(size=4, symbol=symbol),
                            ))

                    _add(ndvi_a, "NDVI",       "NDVI — A",       "#2ca02c")
                    _add(ndvi_b, "NDVI",       "NDVI — B",       "#2ca02c", "dot", "diamond")
                    _add(lst_a,  "LST_C",      "LST °C — A",     "#d62728")
                    _add(lst_b,  "LST_C",      "LST °C — B",     "#d62728", "dot", "diamond")

                    # Rainfall on secondary y-axis (bar)
                    for df_r, period_name, opacity in [
                        (rain_a, "Rainfall A", 0.6),
                        (rain_b, "Rainfall B", 0.35),
                    ]:
                        if not df_r.empty:
                            fig.add_trace(go.Bar(
                                x=df_r["date"], y=df_r["Rainfall_mm"],
                                name=period_name, yaxis="y2",
                                opacity=opacity,
                                marker_color="#1f77b4",
                            ))

                    fig.update_layout(
                        height=380,
                        yaxis  = dict(title="NDVI / LST (°C)", side="left"),
                        yaxis2 = dict(title="Rainfall (mm)", side="right",
                                      overlaying="y", showgrid=False),
                        margin=dict(t=20, b=20, l=60, r=60),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.06, font_size=11),
                        plot_bgcolor="#f8f9fa",
                        barmode="overlay",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Left axis: NDVI (green) + Land Surface Temp (red)   |   "
                        "Right axis: Rainfall bars (blue)   |   "
                        "Solid = Period A  ·  Dotted = Period B"
                    )

                # ── GPS Analytics tab ──────────────────────────────────────────
                elif tab_name == "GPS Analytics":
                    if gps_df is None:
                        st.info("No GPS data loaded.")
                        continue

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        counts = gps_df["animal_id"].value_counts().reset_index()
                        counts.columns = ["animal_id","fixes"]
                        fig = px.bar(counts, x="animal_id", y="fixes",
                                     color="animal_id",
                                     color_discrete_sequence=ANIMAL_COLOURS,
                                     title="Fixes per Animal",
                                     labels={"animal_id":"","fixes":"Fixes"})
                        fig.update_layout(showlegend=False, height=280,
                                          margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        gps_df["date"] = gps_df["timestamp"].dt.date
                        daily = (gps_df.groupby(["date","animal_id"])
                                 .size().reset_index(name="fixes"))
                        fig = px.line(daily, x="date", y="fixes", color="animal_id",
                                      color_discrete_sequence=ANIMAL_COLOURS,
                                      title="Daily Fix Rate",
                                      labels={"date":"","fixes":"Fixes"})
                        fig.update_layout(height=280, margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with c3:
                        gps_df["hour"] = gps_df["timestamp"].dt.hour
                        hc = gps_df.groupby("hour").size().reset_index(name="fixes")
                        fig = px.bar(hc, x="hour", y="fixes",
                                     color="fixes",
                                     color_continuous_scale=["#caf0f8","#0077b6"],
                                     title="Activity by Hour (UTC)",
                                     labels={"hour":"Hour","fixes":"Fixes"})
                        fig.update_layout(coloraxis_showscale=False, height=280,
                                          margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("GPS Data Table"):
                        st.dataframe(
                            gps_df[["animal_id","timestamp","latitude","longitude"]]
                            .reset_index(drop=True),
                            use_container_width=True, height=260,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    gee_ok = init_gee(KEY_PATH, PROJECT_ID)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛰️ Conservation Monitor")
        st.caption("Conservation Platform · Laikipia, Kenya")
        st.divider()

        st.markdown("### View Mode")
        mode    = st.radio("", ["Single period", "Compare two periods"],
                           index=0, horizontal=True)
        compare = (mode == "Compare two periods")
        st.divider()

        st.markdown(
            "<div class='period-header period-a'>Period A</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            start_a = st.date_input("Start A", value=date(2023, 1, 1),
                                    min_value=date(2000, 1, 1),
                                    max_value=date.today(), key="sa")
        with c2:
            end_a   = st.date_input("End A", value=date(2023, 3, 31),
                                    min_value=date(2000, 1, 1),
                                    max_value=date.today(), key="ea")
        if end_a <= start_a:
            end_a = start_a + timedelta(days=30)

        start_b = end_b = None
        if compare:
            st.markdown(
                "<div class='period-header period-b'>Period B</div>",
                unsafe_allow_html=True,
            )
            c3, c4 = st.columns(2)
            with c3:
                start_b = st.date_input("Start B", value=date(2023, 4, 1),
                                        min_value=date(2000, 1, 1),
                                        max_value=date.today(), key="sb")
            with c4:
                end_b   = st.date_input("End B", value=date(2023, 6, 30),
                                        min_value=date(2000, 1, 1),
                                        max_value=date.today(), key="eb")
            if end_b <= start_b:
                end_b = start_b + timedelta(days=30)

        st.divider()
        st.markdown("### 🌍 Satellite Layers")
        if not gee_ok:
            st.warning("Satellite data offline")
        active_layers = st.multiselect(
            "Select layers",
            options=list(LAYER_LABELS.keys()),
            format_func=lambda k: LAYER_LABELS[k],
            default=["NDVI", "Rainfall"],
            disabled=not gee_ok,
        )
        opacity = st.slider("Layer opacity", 0.1, 1.0, 0.75, 0.05)

        st.divider()
        st.markdown("### 🐘 GPS Tracks")
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="animal_id, timestamp, latitude, longitude",
        )
        if uploaded:
            gps_df = load_gps(uploaded)
        elif SAMPLE_CSV.exists():
            gps_df = load_gps(SAMPLE_CSV)
            st.info(f"Sample: {SAMPLE_CSV.name}")
        else:
            gps_df = None

        live_tracking  = st.toggle("🔴 Live Tracking Mode", value=False,
                                    help="Replaces static markers with an animated time slider")
        show_lines     = True
        show_points    = True
        playback_speed = 5
        if live_tracking:
            playback_speed = st.slider("Playback speed", 1, 20, 5)
        else:
            show_lines  = st.checkbox("Track lines", value=True)
            show_points = st.checkbox("GPS fixes",   value=True)

        if gps_df is not None:
            all_animals      = sorted(gps_df["animal_id"].unique().tolist())
            selected_animals = st.multiselect("Animals", all_animals,
                                              default=all_animals)
        else:
            selected_animals = []

        st.divider()
        st.markdown("### 🗺️ Base Map")
        basemap    = st.selectbox(
            "Tiles",
            ["CartoDB Positron","CartoDB Dark Matter","OpenStreetMap","ESRI Satellite"],
            index=0,
        )
        map_height = st.slider("Map height (px)", 400, 900, 600, 20)

    # ── String dates ──────────────────────────────────────────────────────────
    str_a = start_a.strftime("%Y-%m-%d")
    ste_a = end_a.strftime("%Y-%m-%d")
    str_b = start_b.strftime("%Y-%m-%d") if start_b else None
    ste_b = end_b.strftime("%Y-%m-%d")   if end_b   else None

    # ── HEADER ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            "<h1 style='margin-bottom:0'>🛰️ Laikipia Conservation Monitor</h1>"
            "<p style='color:#666;margin-top:2px;font-size:.9rem'>"
            "Live Earth Engine imagery · Laikipia, Kenya</p>",
            unsafe_allow_html=True,
        )
    with h2:
        if gee_ok:
            st.success("Satellite data connected", icon="✅")
        else:
            st.error("Satellite data offline",     icon="❌")

    # KPIs — compact inline cards
    animals_fixes = (
        f"{gps_df['animal_id'].nunique()} / {len(gps_df):,}" if gps_df is not None else "— / —"
    )
    # flex hints: period cards get more room, others shrink to content
    kpi_items = [
        ("Mode",            "Compare" if compare else "Single",  "0 0 auto"),
        ("Period A",        f"{str_a} → {ste_a}",                "1 1 160px"),
        ("Period B",        f"{str_b} → {ste_b}" if compare else "—", "1 1 160px"),
        ("Active Layers",   str(len(active_layers)),             "0 0 auto"),
        ("Animals / Fixes", animals_fixes,                       "0 0 auto"),
    ]
    kpi_html = "".join(
        f"""<div style='flex:{flex};min-width:0;background:#f0f4f8;border-radius:8px;
                        padding:5px 10px;border-top:3px solid #1565c0'>
              <div style='font-size:.6rem;color:#777;font-weight:700;font-family:sans-serif;
                          text-transform:uppercase;letter-spacing:.6px;
                          white-space:nowrap'>{lbl}</div>
              <div style='font-size:.82rem;font-weight:700;color:#0d1b2a;
                          font-family:"Inter","Segoe UI",sans-serif;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{val}</div>
            </div>"""
        for lbl, val, flex in kpi_items
    )
    st.markdown(
        f"<div style='display:flex;gap:8px;margin-bottom:4px;align-items:stretch'>{kpi_html}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MAP AREA  — left drawer + map side-by-side
    # ══════════════════════════════════════════════════════════════════════════
    drawer_open = st.session_state.drawer_open

    col_drawer = None  # defined only when drawer is open

    if not compare:
        # ── Single period map ──────────────────────────────────────────────────
        if drawer_open:
            col_map, col_drawer = st.columns([3, 1])
        else:
            col_map = st.columns([1])[0]  # full-width single column

        with col_map:
            with st.spinner("Building map..."):
                m, layer_cache_a = build_map(
                    active_layers, str_a, ste_a, opacity,
                    basemap, gps_df, show_lines, show_points,
                    selected_animals, map_height,
                    live_tracking=live_tracking,
                    playback_speed=playback_speed,
                )
            map_data = st_folium(
                m, key="map", width=None, height=map_height,
                returned_objects=["last_clicked"],
            )

        # Capture click → open drawer
        if map_data and map_data.get("last_clicked"):
            lc = map_data["last_clicked"]
            st.session_state.clicked_lat  = round(lc["lat"], 5)
            st.session_state.clicked_lon  = round(lc["lng"], 5)
            st.session_state.clicked_from = "map"
            if not st.session_state.drawer_open:
                st.session_state.drawer_open = True
                st.rerun()

        layer_cache_b = {}

    else:
        # ── Compare mode — two maps ────────────────────────────────────────────
        if drawer_open:
            col_maps, col_drawer = st.columns([3, 1])
        else:
            col_maps = st.columns([1])[0]  # full-width

        with col_maps:
            mc_a, mc_b = st.columns(2)
            with mc_a:
                st.markdown(
                    "<div class='period-header period-a' style='font-size:.85rem'>"
                    f"Period A  ·  {str_a} to {ste_a}</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Loading Period A..."):
                    m_a, layer_cache_a = build_map(
                        active_layers, str_a, ste_a, opacity,
                        basemap, gps_df, show_lines, show_points,
                        selected_animals, map_height, label="A",
                        live_tracking=live_tracking,
                        playback_speed=playback_speed,
                    )
                map_data_a = st_folium(m_a, key="map_a", width=None,
                                       height=map_height,
                                       returned_objects=["last_clicked"])

            with mc_b:
                st.markdown(
                    "<div class='period-header period-b' style='font-size:.85rem'>"
                    f"Period B  ·  {str_b} to {ste_b}</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Loading Period B..."):
                    m_b, layer_cache_b = build_map(
                        active_layers, str_b, ste_b, opacity,
                        basemap, gps_df, show_lines, show_points,
                        selected_animals, map_height, label="B",
                        live_tracking=live_tracking,
                        playback_speed=playback_speed,
                    )
                map_data_b = st_folium(m_b, key="map_b", width=None,
                                       height=map_height,
                                       returned_objects=["last_clicked"])

        # Capture click from either map
        last_click = None
        if map_data_a and map_data_a.get("last_clicked"):
            last_click = map_data_a["last_clicked"]
        elif map_data_b and map_data_b.get("last_clicked"):
            last_click = map_data_b["last_clicked"]

        if last_click:
            st.session_state.clicked_lat  = round(last_click["lat"], 5)
            st.session_state.clicked_lon  = round(last_click["lng"], 5)
            st.session_state.clicked_from = "compare"
            if not st.session_state.drawer_open:
                st.session_state.drawer_open = True
                st.rerun()

    # ── Render the left drawer (must be inside the column defined above) ──────
    if drawer_open and col_drawer is not None and st.session_state.clicked_lat is not None:
        with col_drawer:
            _render_left_drawer(
                gee_ok, active_layers,
                layer_cache_a, layer_cache_b if compare else None,
                gps_df,
                str_a, ste_a, str_b, ste_b,
                map_height=map_height,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # BOTTOM CHART DRAWER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    _render_bottom_drawer(
        gee_ok, gps_df,
        str_a, ste_a, str_b, ste_b,
    )



if __name__ == "__main__":
    main()
