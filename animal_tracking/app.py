"""
Animal Tracking Dashboard
=========================
Animated GPS tracking with a playable time slider, per-fix environmental
context from Google Earth Engine, and track analytics.

Run
---
    conservation_env\\Scripts\\streamlit.exe run animal_tracking\\app.py
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import ee
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MiniMap, MousePosition, TimestampedGeoJson
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Animal Tracker · Laikipia",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"]     { background:#0d1b2a; }
  section[data-testid="stSidebar"] *   { color:#e0e0e0 !important; }
  section[data-testid="stSidebar"] hr  { border-color:#2a4a6b; }

  .info-header {
    background:#0d1b2a; color:white; padding:9px 14px;
    font-weight:700; font-size:.88rem; border-radius:12px 12px 0 0;
  }
  .info-body {
    border:1px solid #e0e0e0; border-top:none;
    border-radius:0 0 12px 12px; padding:12px;
    overflow-y:auto;
  }
  .fix-row {
    background:#f8f9fa; border-radius:6px; padding:7px 10px;
    margin-bottom:5px; border-left:3px solid #1565c0;
  }
  .fix-label { font-size:.68rem; color:#666; font-weight:600;
               text-transform:uppercase; letter-spacing:.4px; }
  .fix-value { font-weight:700; color:#0d1b2a; font-size:.95rem; }
  .section-title {
    font-size:.68rem; font-weight:700; color:#555;
    text-transform:uppercase; letter-spacing:.8px; margin:10px 0 4px;
  }
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
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
KEY_PATH   = ROOT / "covariate_extractor" / "secrets" / "gee_service_account.json"
SAMPLE_CSV = ROOT / "covariate_extractor" / "sample_fixes.csv"
PROJECT_ID = "ee-kimanipaul21"

ANIMAL_COLOURS  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
LAIKIPIA_CENTER = [0.3, 37.0]
LAIKIPIA_ZOOM   = 9


# ── GEE init ───────────────────────────────────────────────────────────────────
@st.cache_resource
def init_gee() -> bool:
    try:
        if KEY_PATH.exists():
            info  = json.loads(KEY_PATH.read_text())
            creds = ee.ServiceAccountCredentials(
                email=info["client_email"], key_file=str(KEY_PATH)
            )
            ee.Initialize(credentials=creds, project=PROJECT_ID)
        else:
            ee.Initialize(project=PROJECT_ID)
        return True
    except Exception:
        return False


# ── GPS loading ────────────────────────────────────────────────────────────────
def load_gps(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = df.columns.str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return df.sort_values(["animal_id", "timestamp"]).reset_index(drop=True)


# ── GEE point sampler ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def sample_conditions(lat: float, lon: float, ts_iso: str) -> dict:
    """Sample NDVI, 7-day Rainfall and LST for the week around a GPS fix."""
    try:
        dt    = pd.to_datetime(ts_iso, utc=True)
        start = (dt - timedelta(days=8)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=8)).strftime("%Y-%m-%d")
        pt    = ee.Geometry.Point([lon, lat])

        def safe(img, band, scale, transform=None):
            try:
                v = img.sample(region=pt, scale=scale, numPixels=1).first().get(band).getInfo()
                if v is None:
                    return None
                v = float(v)
                return transform(v) if transform else v
            except Exception:
                return None

        ndvi = safe(
            ee.ImageCollection("MODIS/061/MOD13A1").filterDate(start, end).select("NDVI").mean(),
            "NDVI", 500, lambda v: round(v * 0.0001, 4)
        )
        rain = safe(
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).select("precipitation").sum(),
            "precipitation", 5566, lambda v: round(v, 2)
        )
        lst = safe(
            ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end).select("LST_Day_1km").mean(),
            "LST_Day_1km", 1000, lambda v: round(v * 0.02 - 273.15, 2)
        )
        return {"NDVI": ndvi, "Rainfall_mm": rain, "LST_C": lst}
    except Exception:
        return {}


# ── Haversine distance (km) ────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Map builder ────────────────────────────────────────────────────────────────
def build_tracking_map(
    gps_df: pd.DataFrame,
    selected_animals: list[str],
    basemap: str,
    map_height: int,
    show_tracks: bool,
    speed: int,
) -> folium.Map:

    tiles_cfg = {
        "CartoDB Dark Matter": "CartoDB dark_matter",
        "CartoDB Positron":    "CartoDB positron",
        "OpenStreetMap":       "OpenStreetMap",
        "ESRI Satellite": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri World Imagery",
        ),
    }
    tile = tiles_cfg[basemap]
    if isinstance(tile, tuple):
        m = folium.Map(location=LAIKIPIA_CENTER, zoom_start=LAIKIPIA_ZOOM,
                       tiles=None, control_scale=True)
        folium.TileLayer(tiles=tile[0], attr=tile[1], name=basemap).add_to(m)
    else:
        m = folium.Map(location=LAIKIPIA_CENTER, zoom_start=LAIKIPIA_ZOOM,
                       tiles=tile, control_scale=True)

    filtered = gps_df[gps_df["animal_id"].isin(selected_animals)].copy()
    cmap     = {a: ANIMAL_COLOURS[i % len(ANIMAL_COLOURS)]
                for i, a in enumerate(sorted(filtered["animal_id"].unique()))}

    # Static dashed track lines
    if show_tracks:
        for animal in selected_animals:
            sub    = filtered[filtered["animal_id"] == animal]
            coords = list(zip(sub["latitude"], sub["longitude"]))
            if len(coords) > 1:
                folium.PolyLine(
                    coords, color=cmap[animal], weight=1.8,
                    opacity=0.45, dash_array="6 4",
                    tooltip=f"{animal} full track",
                ).add_to(m)

    # Start / last-fix markers
    for animal in selected_animals:
        sub = filtered[filtered["animal_id"] == animal]
        if sub.empty:
            continue
        first, last = sub.iloc[0], sub.iloc[-1]
        folium.CircleMarker(
            [first["latitude"], first["longitude"]],
            radius=7, color=cmap[animal], fill=True, fill_color="#fff",
            fill_opacity=1, weight=2,
            tooltip=f"{animal} — start ({first['timestamp'].strftime('%Y-%m-%d')})",
        ).add_to(m)
        folium.CircleMarker(
            [last["latitude"], last["longitude"]],
            radius=7, color=cmap[animal], fill=True,
            fill_color=cmap[animal], fill_opacity=1, weight=2,
            tooltip=f"{animal} — last fix ({last['timestamp'].strftime('%Y-%m-%d')})",
        ).add_to(m)

    # Animated markers via TimestampedGeoJson
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
                "time":   ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "popup": (
                    f"<b>{animal}</b><br>"
                    f"🕐 {ts.strftime('%Y-%m-%d %H:%M')}<br>"
                    f"📍 {row['latitude']:.4f}, {row['longitude']:.4f}"
                ),
                "icon": "circle",
                "iconstyle": {
                    "fillColor":   cmap[animal],
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

    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MousePosition(position="topright", prefix="Lat/Lon: ", separator=" | ").add_to(m)

    # Colour key legend injected as fixed HTML
    legend_items = "".join(
        f"<div style='display:flex;align-items:center;margin-bottom:4px'>"
        f"<div style='width:12px;height:12px;border-radius:50%;"
        f"background:{cmap[a]};margin-right:6px;flex-shrink:0'></div>"
        f"<span style='font-size:11px'>{a}</span></div>"
        for a in sorted(cmap)
    )
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;left:10px;bottom:90px;z-index:9999;"
        f"background:rgba(255,255,255,0.93);padding:8px 12px;"
        f"border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.2);"
        f"font-family:sans-serif'>"
        f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.5px;color:#444;margin-bottom:6px'>Animals</div>"
        f"{legend_items}</div>"
    ))

    return m


# ── Info panel HTML helpers ────────────────────────────────────────────────────
def _row(label: str, value: str) -> str:
    return (
        f"<div class='fix-row'>"
        f"<div class='fix-label'>{label}</div>"
        f"<div class='fix-value'>{value}</div>"
        f"</div>"
    )

def _section(title: str) -> str:
    return f"<div class='section-title'>{title}</div>"


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    gee_ok = init_gee()

    # ── Initialise session state ───────────────────────────────────────────────
    for k, v in {
        "panel_open": False,
        "selected_fix": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🐘 Animal Tracker")
        st.caption("Live GPS Tracking · Laikipia, Kenya")
        st.divider()

        st.markdown("### 📂 GPS Data")
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Required columns: animal_id, timestamp, latitude, longitude",
        )
        if uploaded:
            gps_df = load_gps(uploaded)
        elif SAMPLE_CSV.exists():
            gps_df = load_gps(SAMPLE_CSV)
            st.info(f"Sample: {SAMPLE_CSV.name}")
        else:
            gps_df = None
            st.warning("Upload a GPS CSV to begin.")

        if gps_df is None:
            st.stop()

        st.divider()
        st.markdown("### 🎯 Filters")

        min_ts = gps_df["timestamp"].min().date()
        max_ts = gps_df["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_ts, max_ts),
            min_value=min_ts, max_value=max_ts,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            s_d, e_d = date_range
            gps_df = gps_df[
                (gps_df["timestamp"].dt.date >= s_d) &
                (gps_df["timestamp"].dt.date <= e_d)
            ]

        all_animals      = sorted(gps_df["animal_id"].unique().tolist())
        selected_animals = st.multiselect("Animals", all_animals, default=all_animals)

        st.divider()
        st.markdown("### ⚙️ Map Settings")
        basemap     = st.selectbox(
            "Base map",
            ["CartoDB Dark Matter", "CartoDB Positron",
             "OpenStreetMap", "ESRI Satellite"],
        )
        map_height  = st.slider("Map height (px)", 400, 900, 660, 20)
        show_tracks = st.checkbox("Show full track lines", value=True)
        speed       = st.slider("Max playback speed", 1, 20, 5)

        st.divider()
        if gee_ok:
            st.success("GEE connected", icon="✅")
        else:
            st.warning("GEE offline — conditions unavailable", icon="⚠️")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>🐘 Animal Tracking Dashboard</h1>"
        "<p style='color:#666;margin-top:2px;font-size:.9rem'>"
        "Animated GPS tracks · Environmental context · Laikipia, Kenya</p>",
        unsafe_allow_html=True,
    )

    # KPIs
    filtered = gps_df[gps_df["animal_id"].isin(selected_animals)]
    span     = ""
    if len(filtered):
        t0 = filtered["timestamp"].min()
        t1 = filtered["timestamp"].max()
        span = f"{t0.strftime('%b %d')} → {t1.strftime('%b %d, %Y')}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Animals",     len(selected_animals))
    k2.metric("Total Fixes", f"{len(filtered):,}")
    k3.metric("Date Span",   span or "—")
    k4.metric("GEE Status",  "Connected" if gee_ok else "Offline")
    st.divider()

    # ── Map layout ─────────────────────────────────────────────────────────────
    panel_open = st.session_state.panel_open

    if panel_open:
        col_map, col_info = st.columns([3, 1])
    else:
        col_map  = st.columns([1])[0]
        col_info = None

    with col_map:
        st.markdown(
            "<div style='font-size:.8rem;color:#888;margin-bottom:4px'>"
            "▶ Press <b>Play</b> on the time slider to animate · "
            "Click a marker to inspect environmental conditions</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Building tracking map…"):
            m = build_tracking_map(
                gps_df, selected_animals, basemap, map_height, show_tracks, speed
            )
        map_data = st_folium(
            m, key="tracker", width=None, height=map_height,
            returned_objects=["last_clicked", "last_object_clicked_popup"],
        )

    # Capture click
    last_clicked = (map_data or {}).get("last_clicked")
    if last_clicked:
        st.session_state.selected_fix = last_clicked
        if not st.session_state.panel_open:
            st.session_state.panel_open = True
            st.rerun()

    # ── Right panel ────────────────────────────────────────────────────────────
    if panel_open and col_info and st.session_state.selected_fix:
        with col_info:
            lc  = st.session_state.selected_fix
            lat = round(lc["lat"], 5)
            lon = round(lc["lng"], 5)

            # Nearest fix to the clicked coordinates
            tmp = filtered.copy()
            tmp["_dist"] = tmp.apply(
                lambda r: haversine_km(r["latitude"], r["longitude"], lat, lon),
                axis=1,
            )
            nr  = tmp.nsmallest(1, "_dist").iloc[0]
            ts  = nr["timestamp"]

            # Fetch GEE conditions
            cond = {}
            if gee_ok:
                with st.spinner("Fetching conditions…"):
                    cond = sample_conditions(lat, lon, ts.isoformat())

            # Build panel HTML
            body = _section("GPS Fix")
            body += _row("Animal",    nr["animal_id"])
            body += _row("Timestamp", ts.strftime("%Y-%m-%d %H:%M UTC"))
            body += _row("Latitude",  f"{lat:.5f}°")
            body += _row("Longitude", f"{lon:.5f}°")

            body += _section("Environmental Conditions")
            if cond:
                ndvi = cond.get("NDVI")
                rain = cond.get("Rainfall_mm")
                lst  = cond.get("LST_C")

                # NDVI interpretation
                ndvi_txt = "—"
                if ndvi is not None:
                    interp = ("Bare/sparse" if ndvi < 0.1
                              else "Low vegetation" if ndvi < 0.3
                              else "Moderate vegetation" if ndvi < 0.5
                              else "Dense vegetation")
                    ndvi_txt = f"{ndvi:.4f}  ({interp})"

                body += _row("NDVI",      ndvi_txt)
                body += _row("Rainfall",  f"{rain:.1f} mm (7-day)" if rain is not None else "—")
                body += _row("Land Temp", f"{lst:.1f} °C" if lst is not None else "—")
                body += _row("Period",    f"±8 days around fix")
            elif gee_ok:
                body += "<div style='color:#888;font-size:.8rem;padding:6px'>No GEE data for this location.</div>"
            else:
                body += "<div style='color:#888;font-size:.8rem;padding:6px'>GEE offline.</div>"

            # Compute step stats if not first fix
            tmp2 = filtered[filtered["animal_id"] == nr["animal_id"]].sort_values("timestamp")
            idx  = tmp2.index.get_loc(nr.name) if nr.name in tmp2.index else None
            if idx and idx > 0:
                prev = tmp2.iloc[idx - 1]
                d_km  = haversine_km(prev["latitude"], prev["longitude"], lat, lon)
                d_hrs = (ts - prev["timestamp"]).total_seconds() / 3600
                speed_kmh = d_km / d_hrs if d_hrs > 0 else 0
                body += _section("Movement from Previous Fix")
                body += _row("Distance",  f"{d_km:.2f} km")
                body += _row("Time gap",  f"{d_hrs:.1f} h")
                body += _row("Speed",     f"{speed_kmh:.2f} km/h")

            scroll_h = map_height - 48
            st.markdown(
                f"<div style='height:{scroll_h}px;overflow-y:auto'>"
                f"<div class='info-header'>📍 Fix Inspector</div>"
                f"<div class='info-body'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.button(
                "✕ Close", key="close_panel", use_container_width=True,
                on_click=lambda: st.session_state.update(panel_open=False, selected_fix=None),
            )

    # ── Bottom analytics drawer ────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    with st.expander("📊  Track Analytics", expanded=False):
        if filtered.empty or not selected_animals:
            st.info("No data selected.")
        else:
            t_timeline, t_hours, t_dist, t_speed = st.tabs(
                ["Fix Timeline", "Activity Hours", "Cumulative Distance", "Speed"]
            )

            with t_timeline:
                daily = (
                    filtered.assign(date=filtered["timestamp"].dt.date)
                    .groupby(["date", "animal_id"]).size().reset_index(name="fixes")
                )
                fig = px.line(
                    daily, x="date", y="fixes", color="animal_id",
                    color_discrete_sequence=ANIMAL_COLOURS,
                    title="Daily GPS Fix Rate",
                    labels={"date": "Date", "fixes": "Fixes", "animal_id": "Animal"},
                )
                fig.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with t_hours:
                hc = (
                    filtered.assign(hour=filtered["timestamp"].dt.hour)
                    .groupby(["hour", "animal_id"]).size().reset_index(name="fixes")
                )
                fig = px.bar(
                    hc, x="hour", y="fixes", color="animal_id",
                    color_discrete_sequence=ANIMAL_COLOURS, barmode="group",
                    title="Activity by Hour of Day (UTC)",
                    labels={"hour": "Hour (UTC)", "fixes": "Fixes", "animal_id": "Animal"},
                )
                fig.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with t_dist:
                dist_rows = []
                for animal in selected_animals:
                    sub = filtered[filtered["animal_id"] == animal].sort_values("timestamp")
                    if len(sub) < 2:
                        continue
                    cum = 0.0
                    for i in range(1, len(sub)):
                        r0, r1 = sub.iloc[i - 1], sub.iloc[i]
                        cum += haversine_km(r0["latitude"], r0["longitude"],
                                            r1["latitude"], r1["longitude"])
                        dist_rows.append({
                            "timestamp": r1["timestamp"],
                            "animal_id": animal,
                            "cum_km":    round(cum, 3),
                        })
                if dist_rows:
                    fig = px.line(
                        pd.DataFrame(dist_rows),
                        x="timestamp", y="cum_km", color="animal_id",
                        color_discrete_sequence=ANIMAL_COLOURS,
                        title="Cumulative Distance Travelled (km)",
                        labels={"timestamp": "Time", "cum_km": "Distance (km)",
                                "animal_id": "Animal"},
                    )
                    fig.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need ≥ 2 fixes per animal to compute distance.")

            with t_speed:
                spd_rows = []
                for animal in selected_animals:
                    sub = filtered[filtered["animal_id"] == animal].sort_values("timestamp")
                    if len(sub) < 2:
                        continue
                    for i in range(1, len(sub)):
                        r0, r1 = sub.iloc[i - 1], sub.iloc[i]
                        d_km   = haversine_km(r0["latitude"], r0["longitude"],
                                              r1["latitude"], r1["longitude"])
                        d_hr   = (r1["timestamp"] - r0["timestamp"]).total_seconds() / 3600
                        spd_rows.append({
                            "timestamp": r1["timestamp"],
                            "animal_id": animal,
                            "speed_kmh": round(d_km / d_hr, 3) if d_hr > 0 else 0,
                        })
                if spd_rows:
                    fig = px.line(
                        pd.DataFrame(spd_rows),
                        x="timestamp", y="speed_kmh", color="animal_id",
                        color_discrete_sequence=ANIMAL_COLOURS,
                        title="Movement Speed (km/h)",
                        labels={"timestamp": "Time", "speed_kmh": "Speed (km/h)",
                                "animal_id": "Animal"},
                    )
                    fig.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need ≥ 2 fixes per animal to compute speed.")

    st.caption(
        "Animal Tracking Dashboard · Space for Giants · "
        "GPS: prototype data · Imagery: Google Earth Engine"
    )


if __name__ == "__main__":
    main()
