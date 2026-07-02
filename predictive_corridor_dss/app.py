"""
Predictive Corridor DSS — Adaptive Spatial-Temporal Responses of African
Elephants to Climatic Volatility
================================================================================
MSc Thesis operational prototype, Kenyatta University.

Five-layer architecture:
  1. Secure GEE auth + live environmental ingestion (MODIS LST, CHIRPS, NDVI)
  2. Spatio-temporal processing — Behavioral Lag Tracker + SSF
  3. Temporal forecasting — Today / 3-Day / 7-Day predictive deviation engine
  4. Dual-layer interactive map (live raster + vector overlays)
  5. Tactical early-warning dispatch + divergence analytics

GEE is used when `gee-service-key.json` is present and valid; otherwise the
app falls back to an internally generated synthetic environmental surface so
`streamlit run app.py` always works out-of-the-box, with a clear banner
explaining which data source is active.

Run:
    conservation_env\\Scripts\\streamlit.exe run predictive_corridor_dss\\app.py
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Predictive Corridor DSS",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }

  section[data-testid="stSidebar"]    { background:#0d1b14; }
  section[data-testid="stSidebar"] *  { color:#d8e6dc !important; }
  section[data-testid="stSidebar"] hr { border-color:#234032; }
  .stApp { background:#0f1712; color:#e6efe9; }

  .kpi-bar { display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; align-items:stretch; }
  .kpi-card { flex:0 0 auto; background:#16261d; border-radius:8px;
              padding:7px 16px; border-top:3px solid #52b788; }
  .kpi-label { font-size:.62rem; color:#8fae9b; font-weight:700;
               text-transform:uppercase; letter-spacing:.7px; white-space:nowrap; }
  .kpi-value { font-size:1.05rem; font-weight:800; color:#fff; white-space:nowrap; }

  .section-header { font-size:.82rem; font-weight:800; color:#52b788;
                     text-transform:uppercase; letter-spacing:.8px; margin:10px 0 6px; }

  .alert-critical {
    background:linear-gradient(135deg,#4a0e0e,#2b0505); border:1px solid #e63946;
    border-radius:10px; padding:14px 18px; margin-bottom:12px;
  }
  .alert-ok {
    background:#11251a; border:1px solid #2a4a39; border-radius:10px;
    padding:14px 18px; margin-bottom:12px;
  }
  .alert-title { font-weight:800; font-size:.85rem; letter-spacing:.5px; margin-bottom:4px; }
  .alert-body  { font-size:.82rem; color:#dcdcdc; line-height:1.5; }

  div[data-testid="stExpander"] { background:#16261d; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ROOT     = Path(__file__).parent.parent
KEY_PATH = ROOT / "covariate_extractor" / "secrets" / "gee_service_account.json"
LOCAL_KEY_PATH = Path(__file__).parent / "gee-service-key.json"
PROJECT_ID = "ee-kimanipaul21"

# Laikipia–Samburu ecosystem bounding box — used for the AOI, risk grid & telemetry
BBOX = {"min_lon": 36.70, "max_lon": 37.55, "min_lat": 0.10, "max_lat": 1.20}

# Broad Africa extent for the synthetic fallback overlay — no hard clip at
# Kenya's border, matching the continuous pan/zoom feel of live GEE TileLayers
# which stream globally with no spatial bounds.
COUNTRY_BBOX = {"min_lon": -20.0, "max_lon": 55.0, "min_lat": -40.0, "max_lat": 40.0}

BULLS = ["Bull_01", "Bull_02", "Bull_03"]
BULL_COLOURS = {"Bull_01": "#52b788", "Bull_02": "#4895ef", "Bull_03": "#f3722c"}

WATER_POINTS = [(36.95, 0.30), (37.25, 0.85), (37.05, 0.55)]
FENCE_LINE = [(37.30, 0.20), (37.35, 0.45), (37.30, 0.70), (37.20, 0.95)]   # smart-fence vector
COMMUNITY_FARMS = [(37.32, 0.25, "Sector C4"), (37.28, 0.60, "Sector C7"), (37.18, 0.92, "Sector C9")]

HIST_DAYS = 35           # total simulated telemetry window: 14 quiet lead-in + 21-day shock cycle
MAX_FORECAST_DAYS = 14   # furthest the predictive engine will project

def scenario_for_days_out(days_out: float) -> dict:
    """
    Continuous escalation of thermal stress / rainfall decay / NDVI shock as a
    function of how far into the future we're projecting — replaces fixed
    Today/3-Day/7-Day buckets with a smooth function so any date can be picked.
    """
    f = max(0.0, min(days_out, MAX_FORECAST_DAYS)) / MAX_FORECAST_DAYS
    return {
        "days_out": days_out,
        "temp_bump": 4.0 * f,
        "rain_decay": 1.0 - 0.70 * f,
        "ndvi_shock": 1.0 + 1.20 * f,
    }

RISK_PALETTE = [
    "#ffffb2", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
]

# Scientific colour ramps for the environmental index layers — distinct from
# RISK_PALETTE (reserved for the connectivity-risk grid) so temperature is
# never visually conflated with "risk". Ranges/palettes mirror the validated
# visParams used in the companion gee_map_viewer project (practical
# land-cover bounds, not the theoretical index extremes).
NDVI_PALETTE = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]  # -0.1 → 0.9
LST_PALETTE = ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]              # 15°C → 50°C, diverging thermal
RAIN_PALETTE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b"]  # 0 → 300 mm (10-day cumulative)

SEED = 42


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def point_to_segment_km(plon, plat, alon, alat, blon, blat) -> float:
    """Shortest distance from point P to segment AB, in km (planar approx, fine at this scale)."""
    ax, ay, bx, by, px, py = alon, alat, blon, blat, plon, plat
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return haversine_km(plon, plat, alon, alat)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nlon, nlat = ax + t * dx, ay + t * dy
    return haversine_km(plon, plat, nlon, nlat)


def dist_to_fence_km(plon: float, plat: float) -> float:
    return min(
        point_to_segment_km(plon, plat, *FENCE_LINE[i], *FENCE_LINE[i + 1])
        for i in range(len(FENCE_LINE) - 1)
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SECURE GEE AUTH & LIVE ENVIRONMENTAL INGESTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to Earth Engine…")
def init_gee() -> tuple[bool, str]:
    """
    Secure GEE initialisation via ee.ServiceAccountCredentials, reading the key
    from a local 'gee-service-key.json' (or the shared covariate_extractor key
    as a fallback during local development). Returns (success, message).
    """
    try:
        import ee

        key_path = LOCAL_KEY_PATH if LOCAL_KEY_PATH.exists() else KEY_PATH
        if not key_path.exists():
            if "gee" in st.secrets:
                info = dict(st.secrets["gee"])
                if "private_key" in info:
                    info["private_key"] = info["private_key"].replace("\\n", "\n")
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_info(
                    info, scopes=["https://www.googleapis.com/auth/earthengine"]
                )
                ee.Initialize(credentials=creds, project=PROJECT_ID)
                ee.Number(1).getInfo()
                return True, "Connected via Streamlit secrets."
            return False, f"No GEE key found at '{key_path.name}'."

        info = json.loads(key_path.read_text(encoding="utf-8"))
        creds = ee.ServiceAccountCredentials(info["client_email"], str(key_path))
        ee.Initialize(creds, project=PROJECT_ID)
        ee.Number(1).getInfo()
        return True, "Satellite imagery connected."
    except Exception as exc:
        return False, f"GEE auth failed: {exc}"


@st.cache_data(ttl=3600, show_spinner="Streaming live MODIS / CHIRPS / NDVI…")
def fetch_live_environmental_layers(gee_ready: bool, ref_date: str) -> dict:
    """
    Pull live LST, precipitation and NDVI-anomaly map tiles + bbox-mean stats
    from GEE. On any failure (no GEE, quota, network) falls back to an
    internally generated synthetic surface so the dashboard never breaks.
    """
    if gee_ready:
        try:
            import ee

            region = ee.Geometry.Rectangle(
                [BBOX["min_lon"], BBOX["min_lat"], BBOX["max_lon"], BBOX["max_lat"]]
            )
            end = ee.Date(ref_date)

            # Product-appropriate lookback windows:
            # Product-appropriate lookback windows account for NASA/UCSB
            # processing latency (typically 4-6 weeks behind real-time):
            # MOD13A1 (16-day composite) → 60 days
            # MOD11A2 (8-day composite)  → 16 days
            # CHIRPS DAILY               → 40 days (also ~5-week lag)
            ndvi_start  = end.advance(-60, "day")
            lst_start   = end.advance(-16, "day")
            rain_start  = end.advance(-40, "day")

            ndvi_img = (
                ee.ImageCollection("MODIS/061/MOD13A1")
                .filterDate(ndvi_start, end)
                .select("NDVI")
                .map(lambda i: i.multiply(0.0001))
                .median()
                .rename("NDVI")
            )
            lst_img = (
                ee.ImageCollection("MODIS/061/MOD11A2")
                .filterDate(lst_start, end)
                .select("LST_Day_1km")
                .map(lambda i: i.multiply(0.02).subtract(273.15).rename("LST_C"))
                .mean()
            )
            chirps_img = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(rain_start, end)
                .select("precipitation")
                .sum()
                .rename("rainfall_mm")
            )
            ndvi_anom_img = ndvi_img.subtract(0.45)  # crude anomaly vs long-term mean proxy

            lst_mapid  = lst_img.getMapId({"min": 15, "max": 50, "palette": LST_PALETTE})
            ndvi_mapid = ndvi_img.getMapId({"min": -0.1, "max": 0.9, "palette": NDVI_PALETTE})
            rain_mapid = chirps_img.getMapId({"min": 0, "max": 300, "palette": RAIN_PALETTE})

            # bands already named "NDVI", "LST_C", "rainfall_mm" from the
            # image constructors above — just rename to short stat keys
            stats = (
                ndvi_img.rename("ndvi")
                .addBands(ndvi_anom_img.rename("ndvi_anom"))
                .addBands(lst_img.rename("lst"))
                .addBands(chirps_img.rename("rain"))
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=2000, bestEffort=True)
                .getInfo()
            )
            return {
                "source": "live",
                "ref_date": ref_date,
                "lst_tile_url": lst_mapid["tile_fetcher"].url_format,
                "ndvi_tile_url": ndvi_mapid["tile_fetcher"].url_format,
                "rain_tile_url": rain_mapid["tile_fetcher"].url_format,
                "mean_lst_c": float(stats.get("lst") or 30.0),
                "mean_rain_mm": float(stats.get("rain") or 5.0),
                "mean_ndvi": float(stats.get("ndvi") or 0.3),
                "mean_ndvi_anom": float(stats.get("ndvi_anom") or -0.1),
            }
        except Exception as exc:
            st.session_state["_gee_fetch_error"] = str(exc)

    # ── Synthetic fallback surface (deterministic, seeded) ──────────────────
    rng = np.random.default_rng(abs(hash(ref_date)) % (2**31))
    return {
        "source": "synthetic",
        "ref_date": ref_date,
        "lst_tile_url": None,
        "ndvi_tile_url": None,
        "rain_tile_url": None,
        "mean_lst_c": round(34.0 + rng.normal(0, 1.5), 1),
        "mean_rain_mm": round(max(0.0, rng.normal(35.0, 18.0)), 1),  # mm, 10-day cumulative
        "mean_ndvi": round(float(np.clip(0.3 + rng.normal(-0.15, 0.08), -1, 1)), 3),
        "mean_ndvi_anom": round(float(rng.normal(-0.15, 0.08)), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SPATIO-TEMPORAL PROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Pulling EarthRanger collar feed…")
def mock_earthranger_stream(env: dict, seed: int = SEED) -> pd.DataFrame:
    """
    Programmatically mocked telemetry shaped like an EarthRanger collar feed
    (subject_id / recorded_at / location) for 3 crop-raiding bulls, spanning
    21 days that capture an escalating thermal/NDVI shock. A hidden ground-
    truth lag drives each bull's eventual swing toward community boundaries —
    used downstream to validate the statistical lag detector.
    """
    rng = np.random.default_rng(seed)
    lead_in_days = HIST_DAYS - 21  # quiet baseline period before the shock cycle
    n_days = HIST_DAYS
    n_hours = n_days * 24
    base_date = datetime.combine(date.today() - timedelta(days=n_days), datetime.min.time())
    hourly_idx = pd.date_range(base_date, periods=n_hours, freq="h")

    # daily climate trace anchored to the live/synthetic bbox-mean reading
    tmax = np.zeros(n_days)
    ndvi = np.zeros(n_days)
    val = env["mean_ndvi_anom"]
    for i in range(n_days):
        j = i - lead_in_days  # index into the 21-day shock cycle, negative during lead-in
        if j < 0:
            tmax[i] = env["mean_lst_c"] - 4 + rng.normal(0, 0.4)
            val += rng.normal(0, 0.01)
        elif j < 14:
            tmax[i] = env["mean_lst_c"] - 4 + j * 0.65 + rng.normal(0, 0.4)
            val -= 0.03 + rng.normal(0, 0.01)
        elif j < 16:
            tmax[i] = env["mean_lst_c"] - 8 + rng.normal(0, 0.5)
            val += 0.05 + rng.normal(0, 0.01)
        else:
            tmax[i] = env["mean_lst_c"] - 6 + rng.normal(0, 1.0)
            val += 0.03 + rng.normal(0, 0.01)
        ndvi[i] = np.clip(val, -1.0, 1.0)
    day_of_hour = np.repeat(np.arange(n_days), 24)[:n_hours]

    shock_hits = np.where((tmax > 38.0) & (ndvi < -0.4))[0]
    shock_day = int(shock_hits[0]) if len(shock_hits) else int(np.argmax(tmax))
    shock_hour = shock_day * 24

    records = []
    for bull in BULLS:
        home_lon = 37.0 + rng.uniform(-0.10, 0.10)
        home_lat = 0.50 + rng.uniform(-0.15, 0.15)
        lon, lat = home_lon, home_lat
        bearing = rng.uniform(0, 360)
        lag_true = int(rng.integers(6, 30))
        response_hour = shock_hour + lag_true
        farm_lon, farm_lat = COMMUNITY_FARMS[rng.integers(0, len(COMMUNITY_FARMS))][:2]

        for h in range(n_hours):
            d = day_of_hour[h]
            heat_stress = max(0.0, tmax[d] - 32) / 10.0
            base_step = 0.0022 + heat_stress * 0.0017
            bearing = (bearing + rng.normal(0, 25)) % 360

            if h >= response_hour:
                dlon, dlat = farm_lon - lon, farm_lat - lat
                target_bearing = math.degrees(math.atan2(dlon, dlat)) % 360
                bearing = (0.55 * bearing + 0.45 * target_bearing) % 360
                step = base_step * 2.3
            else:
                step = base_step

            rad = math.radians(bearing)
            lon += step * math.sin(rad)
            lat += step * math.cos(rad)

            records.append({
                "subject_id": bull,
                "recorded_at": hourly_idx[h],
                "hour_idx": h,
                "location": json.dumps({"lon": round(lon, 6), "lat": round(lat, 6)}),
                "lon": round(lon, 6), "lat": round(lat, 6),
                "tmax_c": round(float(tmax[d]), 2),
                "ndvi_anomaly": round(float(ndvi[d]), 3),
                "lag_hours_true": lag_true,
                "response_hour_idx": response_hour,
                "target_farm_lon": farm_lon, "target_farm_lat": farm_lat,
            })

    df = pd.DataFrame(records)
    df["recorded_date"] = df["recorded_at"].dt.date
    df.attrs["shock_day"] = shock_day
    df.attrs["shock_hour"] = shock_hour
    df.attrs["base_date"] = base_date.date()
    df.attrs["climate_daily"] = pd.DataFrame({
        "day": np.arange(n_days),
        "calendar_date": [base_date.date() + timedelta(days=int(i)) for i in range(n_days)],
        "tmax_c": tmax.round(2), "ndvi_anomaly": ndvi.round(3),
    })
    return df


def detect_behavioral_lag(telemetry_df: pd.DataFrame, shock_hour: int) -> pd.DataFrame:
    """
    Behavioral Lag Time: hours between the climatic shock and the first
    sustained (6h rolling mean) breach of a 2.5σ threshold on the rate of
    approach toward the bull's eventual target farm boundary — isolating the
    directional response from heat-driven baseline restlessness noise.
    """
    rows = []
    for bull, grp in telemetry_df.groupby("subject_id"):
        grp = grp.sort_values("hour_idx").reset_index(drop=True)
        farm_lon, farm_lat = grp["target_farm_lon"].iloc[0], grp["target_farm_lat"].iloc[0]
        grp["dist_farm_km"] = grp.apply(
            lambda r: haversine_km(r["lon"], r["lat"], farm_lon, farm_lat), axis=1
        )
        grp["approach_rate"] = -grp["dist_farm_km"].diff().fillna(0.0)

        baseline = grp.loc[grp["hour_idx"] < shock_hour, "approach_rate"]
        mu, sigma = baseline.mean(), baseline.std(ddof=0) or 1e-6
        threshold = mu + 2.5 * sigma

        post = grp.loc[grp["hour_idx"] >= shock_hour].copy()
        post["roll"] = post["approach_rate"].rolling(6, min_periods=6).mean()
        exceed = post.loc[post["roll"] > threshold]

        lag_detected = int(exceed.iloc[0]["hour_idx"] - shock_hour) if not exceed.empty else np.nan
        rows.append({
            "subject_id": bull,
            "lag_hours_detected": lag_detected,
            "lag_hours_true": int(grp["lag_hours_true"].iloc[0]),
            "baseline_approach_km_h": round(mu, 4),
            "approach_threshold_km_h": round(threshold, 4),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Generating counterfactual steps…")
def generate_steps_with_alternatives(telemetry_df: pd.DataFrame, n_alt: int = 4,
                                      seed: int = SEED) -> pd.DataFrame:
    """4 counterfactual steps per actual step, resampling empirical step-length
    and a Gaussian turn-angle distribution around the previous bearing (SSF design)."""
    rng = np.random.default_rng(seed)
    out = []

    for bull, grp in telemetry_df.groupby("subject_id"):
        grp = grp.sort_values("hour_idx").reset_index(drop=True).iloc[::4].reset_index(drop=True)
        lons, lats, ts = grp["lon"].values, grp["lat"].values, grp["recorded_at"].values

        emp_steps, emp_bearings = [], []
        for i in range(1, len(grp)):
            dlon, dlat = lons[i] - lons[i - 1], lats[i] - lats[i - 1]
            emp_steps.append(math.hypot(dlon, dlat))
            emp_bearings.append(math.degrees(math.atan2(dlon, dlat)) % 360)
        emp_steps = np.array(emp_steps) if emp_steps else np.array([0.001])

        for i in range(1, len(grp)):
            sl, sla = lons[i - 1], lats[i - 1]
            out.append({
                "subject_id": bull, "step_id": i, "type": "actual", "label": 1,
                "start_lon": sl, "start_lat": sla,
                "end_lon": lons[i], "end_lat": lats[i], "timestamp": ts[i],
            })
            prev_bearing = emp_bearings[i - 1]
            for _ in range(n_alt):
                rand_len = rng.choice(emp_steps)
                cf_bearing = (prev_bearing + rng.normal(0, 45)) % 360
                rad = math.radians(cf_bearing)
                out.append({
                    "subject_id": bull, "step_id": i, "type": "counterfactual", "label": 0,
                    "start_lon": sl, "start_lat": sla,
                    "end_lon": sl + rand_len * math.sin(rad),
                    "end_lat": sla + rand_len * math.cos(rad),
                    "timestamp": ts[i],
                })
    return pd.DataFrame(out)


def extract_step_covariates(steps_df: pd.DataFrame, climate_daily: pd.DataFrame,
                             base_date: pd.Timestamp, water_points: list[tuple[float, float]]) -> pd.DataFrame:
    """Sample localized temperature, NDVI and water-distance covariates for each
    actual/counterfactual endpoint — the GEE `ee.Image.sample()` equivalent
    when running against the synthetic fallback surface."""
    df = steps_df.copy()
    df["day_idx"] = ((pd.to_datetime(df["timestamp"]) - base_date).dt.total_seconds() // 86400).astype(int)
    df["day_idx"] = df["day_idx"].clip(0, len(climate_daily) - 1)

    clim = climate_daily.set_index("day")
    df["thermal_stress"] = df["day_idx"].map(clim["tmax_c"]).astype(float)
    df["veg_density"] = df["day_idx"].map(clim["ndvi_anomaly"]).apply(lambda v: (v + 1) / 2).astype(float)
    df["dist_water_km"] = df.apply(
        lambda r: min(haversine_km(r["end_lon"], r["end_lat"], wlon, wlat) for wlon, wlat in water_points),
        axis=1,
    )
    return df


def fit_ssf_logistic(df: pd.DataFrame, features=("thermal_stress", "veg_density", "dist_water_km"),
                      lr: float = 0.15, n_iter: int = 800):
    """Pooled discrete-choice logistic regression (numpy-only) approximating a
    conditional SSF → landscape resistance coefficients (β)."""
    X = df[list(features)].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)

    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])

    beta = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(n_iter):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) / n
        beta -= lr * grad

    table = pd.DataFrame({"covariate": ["Intercept"] + list(features), "beta": beta.round(4)})
    table["interpretation"] = table["beta"].apply(
        lambda b: "Selects despite stress" if b > 0 else "Avoids / stress-driven deviation"
    )
    return table, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — TEMPORAL FORECASTING & PREDICTIVE DEVIATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_baseline_corridor(telemetry_df: pd.DataFrame, bull: str, width_km: float = 1.2) -> list[tuple[float, float]]:
    """
    The 'ancestral memory' Normal Pattern Baseline: a smoothed corridor built
    from the bull's pre-shock movement, used as the reference against which
    climate-driven deviation is measured.
    """
    grp = telemetry_df[telemetry_df["subject_id"] == bull].sort_values("hour_idx")
    shock_hour = telemetry_df.attrs.get("shock_hour", grp["hour_idx"].max() // 2)
    pre_shock = grp[grp["hour_idx"] < shock_hour]
    if len(pre_shock) < 5:
        pre_shock = grp.iloc[: max(5, len(grp) // 3)]
    # thin to ~12 vertices for a clean polyline
    idx = np.linspace(0, len(pre_shock) - 1, min(12, len(pre_shock))).astype(int)
    return list(zip(pre_shock["lon"].values[idx], pre_shock["lat"].values[idx]))


def project_forward_trajectory(telemetry_df: pd.DataFrame, bull: str, days_out: int,
                                temp_bump: float, rain_decay: float, ndvi_shock: float,
                                seed: int = SEED) -> pd.DataFrame:
    """
    Pattern Attrition Engine: projects the bull's trajectory `days_out` days
    into the future, escalating thermal stress / NDVI shock and dampening
    rainfall recovery, then biases the heading toward its drying water /
    target farm proportionally to forecast severity. days_out == 0 returns
    the live position unchanged ("Today").
    """
    days_out = int(round(days_out))
    rng = np.random.default_rng(seed + hash(bull) % 1000)
    grp = telemetry_df[telemetry_df["subject_id"] == bull].sort_values("hour_idx")
    lon, lat = float(grp["lon"].iloc[-1]), float(grp["lat"].iloc[-1])
    farm_lon, farm_lat = grp["target_farm_lon"].iloc[0], grp["target_farm_lat"].iloc[0]
    bearing = rng.uniform(0, 360)

    severity = (temp_bump / 4.0) + (1 - rain_decay) + (ndvi_shock - 1.0)
    pull_strength = float(np.clip(0.15 + severity * 0.25, 0.15, 0.85))

    rows = [{"hour_offset": 0, "lon": lon, "lat": lat}]
    for h in range(1, days_out * 24 + 1):
        bearing = (bearing + rng.normal(0, 20)) % 360
        dlon, dlat = farm_lon - lon, farm_lat - lat
        target_bearing = math.degrees(math.atan2(dlon, dlat)) % 360
        bearing = ((1 - pull_strength) * bearing + pull_strength * target_bearing) % 360

        step = (0.0022 + severity * 0.0028)
        rad = math.radians(bearing)
        lon += step * math.sin(rad)
        lat += step * math.cos(rad)
        rows.append({"hour_offset": h, "lon": lon, "lat": lat})

    df = pd.DataFrame(rows)
    df["subject_id"] = bull
    df["pull_strength"] = pull_strength
    return df


def compute_deviation_metrics(baseline_corridor: list[tuple[float, float]],
                               projected_df: pd.DataFrame) -> dict:
    """
    Behavioral Deviation Metric: lateral distance (km) between the projected
    endpoint and the nearest point on the baseline corridor, plus distance
    to the nearest fence segment and nearest community farm — used to decide
    whether a pre-emptive tactical alert should fire.
    """
    end_lon, end_lat = float(projected_df["lon"].iloc[-1]), float(projected_df["lat"].iloc[-1])

    if len(baseline_corridor) >= 2:
        dists = [
            point_to_segment_km(end_lon, end_lat, *baseline_corridor[i], *baseline_corridor[i + 1])
            for i in range(len(baseline_corridor) - 1)
        ]
        deviation_km = min(dists)
    else:
        deviation_km = 0.0

    fence_km = dist_to_fence_km(end_lon, end_lat)
    farm_dists = [
        (name, haversine_km(end_lon, end_lat, flon, flat))
        for flon, flat, name in COMMUNITY_FARMS
    ]
    nearest_farm, nearest_farm_km = min(farm_dists, key=lambda t: t[1])

    return {
        "deviation_km": round(deviation_km, 2),
        "fence_distance_km": round(fence_km, 2),
        "nearest_farm": nearest_farm,
        "nearest_farm_km": round(nearest_farm_km, 2),
        "end_lon": end_lon, "end_lat": end_lat,
    }


def divergence_series(baseline_corridor: list[tuple[float, float]], telemetry_df: pd.DataFrame,
                       bull: str, temp_bump: float, rain_decay: float, ndvi_shock: float) -> pd.DataFrame:
    """Day-by-day widening gap (km) between ancestral baseline and the climate-stressed
    projected reality, for the dual-axis divergence chart — days 0 through 7."""
    rows = []
    for d in range(0, 8):
        scale = d / 7.0
        proj = project_forward_trajectory(
            telemetry_df, bull, days_out=d,
            temp_bump=temp_bump * scale, rain_decay=1 - (1 - rain_decay) * scale,
            ndvi_shock=1 + (ndvi_shock - 1) * scale,
        )
        metrics = compute_deviation_metrics(baseline_corridor, proj)
        rows.append({
            "day": d,
            "deviation_km": metrics["deviation_km"],
            "fence_distance_km": metrics["fence_distance_km"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# RISK GRID (bottleneck cells — glow red/orange under longer forecast horizons)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_risk_grid(bbox: dict, n_per_side: int = 12) -> pd.DataFrame:
    lons = np.linspace(bbox["min_lon"], bbox["max_lon"], n_per_side + 1)
    lats = np.linspace(bbox["min_lat"], bbox["max_lat"], n_per_side + 1)
    records, cell_id = [], 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            lon0, lon1, lat0, lat1 = lons[i], lons[i + 1], lats[j], lats[j + 1]
            clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
            records.append({
                "cell_id": cell_id, "lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
                "centroid_lon": clon, "centroid_lat": clat,
                "dist_water_km": min(haversine_km(clon, clat, wlon, wlat) for wlon, wlat in WATER_POINTS),
                "dist_fence_km": dist_to_fence_km(clon, clat),
            })
            cell_id += 1
    return pd.DataFrame(records)


def compute_grid_risk(grid_df: pd.DataFrame, timeline_params: dict, beta_table: pd.DataFrame,
                       mu, sigma, env: dict) -> pd.DataFrame:
    """Bottleneck risk per cell — grows and glows hotter as the forecast horizon extends."""
    df = grid_df.copy()
    thermal = env["mean_lst_c"] + timeline_params["temp_bump"]
    veg = np.clip((env["mean_ndvi_anom"] - 0.2 * (timeline_params["ndvi_shock"] - 1)) , -1, 1)
    veg_density = (veg + 1) / 2
    water_penalty = df["dist_water_km"] / max(timeline_params["rain_decay"], 0.05)

    feats = np.column_stack([
        np.full(len(df), thermal),
        np.full(len(df), veg_density),
        water_penalty.to_numpy(),
    ])
    Xs = (feats - mu) / sigma
    beta = beta_table.set_index("covariate")["beta"]
    z = beta["Intercept"] + Xs @ beta[["thermal_stress", "veg_density", "dist_water_km"]].to_numpy()
    selection_p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    fence_proximity_w = 1.0 / (1.0 + df["dist_fence_km"] / 4.0)
    horizon_amplifier = 1.0 + timeline_params["days_out"] / 7.0   # glows hotter further out

    raw = selection_p * fence_proximity_w.to_numpy() * horizon_amplifier
    df["risk_score"] = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 100).round(1)
    df["bottleneck"] = df["risk_score"] >= df["risk_score"].quantile(0.82)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def risk_colour(score: float) -> str:
    idx = min(int(score / 100 * len(RISK_PALETTE)), len(RISK_PALETTE) - 1)
    return RISK_PALETTE[idx]


def synthetic_field(layer_key: str, synthetic_mean: float, value_range: tuple[float, float],
                     px: int = 256) -> np.ndarray:
    """
    Deterministic (seeded) coarse noise field, bilinearly upsampled to a
    smooth px×px raster spanning COUNTRY_BBOX — used both to paint the
    overlay image and to answer point-click queries, so the two always agree.
    """
    seed = abs(hash((layer_key, round(synthetic_mean, 3)))) % (2**31)
    rng = np.random.default_rng(seed)
    lo, hi = value_range
    coarse_n = 48  # larger grid covers the broad Africa bbox without blocky pixels
    lat_lin = np.linspace(COUNTRY_BBOX["max_lat"], COUNTRY_BBOX["min_lat"], coarse_n)
    lon_lin = np.linspace(COUNTRY_BBOX["min_lon"], COUNTRY_BBOX["max_lon"], coarse_n)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
    aoi_clat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    aoi_clon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
    dist_from_aoi = np.hypot(lat_grid - aoi_clat, lon_grid - aoi_clon)
    drift_scale = np.clip(dist_from_aoi * 0.04, 0, (hi - lo) * 0.3)
    coarse_field = synthetic_mean + rng.normal(0, (hi - lo) * 0.12, size=(coarse_n, coarse_n)) \
        + rng.normal(0, 1, size=(coarse_n, coarse_n)) * drift_scale
    coarse_field = np.clip(coarse_field, lo, hi)
    coarse_img = Image.fromarray(coarse_field.astype(np.float32), mode="F")
    return np.array(coarse_img.resize((px, px), resample=Image.BILINEAR))


def sample_synthetic_field_at(layer_key: str, synthetic_mean: float, value_range: tuple[float, float],
                               lat: float, lon: float) -> float:
    """Look up the value of the same field rendered on the map, at a clicked point."""
    field = synthetic_field(layer_key, synthetic_mean, value_range)
    px = field.shape[0]
    row = int(np.clip((COUNTRY_BBOX["max_lat"] - lat) / (COUNTRY_BBOX["max_lat"] - COUNTRY_BBOX["min_lat"]) * (px - 1), 0, px - 1))
    col = int(np.clip((lon - COUNTRY_BBOX["min_lon"]) / (COUNTRY_BBOX["max_lon"] - COUNTRY_BBOX["min_lon"]) * (px - 1), 0, px - 1))
    return float(field[row, col])


@st.cache_data(ttl=300, show_spinner=False)
def sample_live_point(layer_key: str, lat: float, lon: float, ref_date: str) -> float | None:
    """Point-sample the live GEE image for the given layer at a clicked location."""
    try:
        import ee
        point = ee.Geometry.Point([lon, lat])
        end = ee.Date(ref_date)
        if layer_key == "NDVI":
            img = (ee.ImageCollection("MODIS/061/MOD13A1").filterDate(end.advance(-60, "day"), end)
                   .select("NDVI").map(lambda i: i.multiply(0.0001)).median())
        elif layer_key == "LST":
            img = (ee.ImageCollection("MODIS/061/MOD11A2").filterDate(end.advance(-16, "day"), end)
                   .select("LST_Day_1km")
                   .map(lambda i: i.multiply(0.02).subtract(273.15).rename("LST_C")).mean())
        else:
            img = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(end.advance(-40, "day"), end)
                   .select("precipitation").sum())
        value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=1000, bestEffort=True).getInfo()
        return float(next(iter(value.values())))
    except Exception:
        return None


def hex_palette_to_rgba(frac: np.ndarray, palette: list[str], alpha: int = 140) -> np.ndarray:
    """
    Maps a [0,1] float array to RGBA uint8 by linearly interpolating across a
    hex colour palette — a matplotlib-free stand-in for cmap(frac), since
    matplotlib isn't a declared/guaranteed dependency on the deploy target.
    """
    stops = np.linspace(0.0, 1.0, len(palette))
    rgb_stops = np.array([[int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for h in palette])
    flat = frac.ravel()
    r = np.interp(flat, stops, rgb_stops[:, 0])
    g = np.interp(flat, stops, rgb_stops[:, 1])
    b = np.interp(flat, stops, rgb_stops[:, 2])
    rgba = np.stack([r, g, b, np.full_like(r, alpha)], axis=-1).astype(np.uint8)
    return rgba.reshape(*frac.shape, 4)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — DUAL-LAYER INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════════════════════

def build_operational_map(env: dict, telemetry_df: pd.DataFrame, steps_df: pd.DataFrame,
                           grid_df: pd.DataFrame, bulls_shown: list[str],
                           baseline_corridors: dict, projections: dict,
                           show_counterfactual: bool, timeline_label: str,
                           active_layers: set[str] | None = None) -> folium.Map:
    clat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    clon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
    m = folium.Map(location=[clat, clon], zoom_start=9, tiles=None)
    active_layers = active_layers if active_layers is not None else {"NDVI", "LST", "Rainfall"}

    # ── Selectable basemaps (radio group, top-right in-map panel) ───────────
    folium.TileLayer("CartoDB dark_matter", name="Dark Matter", overlay=False, control=True, show=True).add_to(m)
    folium.TileLayer("CartoDB positron", name="Positron (Light)", overlay=False, control=True, show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True, show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)", overlay=False, control=True, show=False,
    ).add_to(m)

    # ── Environmental overlays (NDVI / LST / Rainfall) ──────────────────────
    # Live GEE: TileLayer with z/x/y URLs — globally continuous, no spatial
    # clip, exactly like gee_map_viewer. GEE offline: the checkbox still
    # appears in the layer panel (so the UI is consistent) but no raster is
    # added — same pattern as gee_map_viewer's "satellite data offline" state.
    is_live = env.get("source") == "live"

    def add_env_overlay(layer_key: str, tile_url: str | None, label: str) -> None:
        show = layer_key in active_layers
        if is_live and tile_url:
            folium.raster_layers.TileLayer(
                tiles=tile_url, attr=f"MODIS/CHIRPS via NASA · {label}",
                name=label, overlay=True, opacity=0.55, show=show,
            ).add_to(m)
        else:
            # Placeholder FeatureGroup so the checkbox exists in the layer
            # panel even when GEE is offline — no bounded image, no clip.
            folium.FeatureGroup(name=f"{label} (Satellite offline)", overlay=True, show=False).add_to(m)

    add_env_overlay("NDVI",     env.get("ndvi_tile_url"), "NDVI")
    add_env_overlay("LST",      env.get("lst_tile_url"),  "Land Surface Temp")
    add_env_overlay("Rainfall", env.get("rain_tile_url"), "Rainfall")

    # ── AOI boundary — togglable outline of the Laikipia–Samburu study area ─
    aoi_fg = folium.FeatureGroup(name="AOI Boundary (Laikipia–Samburu)", overlay=True, show=True)
    folium.Rectangle(
        bounds=[[BBOX["min_lat"], BBOX["min_lon"]], [BBOX["max_lat"], BBOX["max_lon"]]],
        color="#52b788", weight=2.5, fill=False, dash_array="6,4",
        tooltip="Area of Interest — Laikipia–Samburu Ecosystem",
    ).add_to(aoi_fg)
    aoi_fg.add_to(m)

    # ── Risk grid (bottleneck cells glow hotter on longer horizons) ─────────
    grid_fg = folium.FeatureGroup(name="Connectivity Risk Grid", overlay=True, show=True)
    for _, row in grid_df.iterrows():
        bounds = [[row["lat0"], row["lon0"]], [row["lat1"], row["lon1"]]]
        is_crit = bool(row["bottleneck"])
        folium.Rectangle(
            bounds=bounds,
            color="#ff3333" if is_crit else "#3a3a3a",
            weight=2.4 if is_crit else 0.3,
            fill=True, fill_color=risk_colour(row["risk_score"]),
            fill_opacity=0.6 if is_crit else 0.35,
            tooltip=(
                f"Cell {row['cell_id']}<br>Risk: <b>{row['risk_score']:.1f}%</b>"
                + ("<br><b>⚠️ Bottleneck — " + timeline_label + "</b>" if is_crit else "")
            ),
        ).add_to(grid_fg)
    grid_fg.add_to(m)

    # ── Baseline corridor (semi-transparent green polygon band) ─────────────
    for bull in bulls_shown:
        corridor = baseline_corridors.get(bull, [])
        if len(corridor) >= 2:
            band = [[lat, lon] for lon, lat in corridor]
            folium.PolyLine(band, color="#2dc653", weight=10, opacity=0.25,
                             tooltip=f"{bull} — Normal Pattern Baseline Corridor").add_to(m)
            folium.PolyLine(band, color="#2dc653", weight=2, opacity=0.8, dash_array="1,6").add_to(m)

    # ── Counterfactual steps (sparse, dashed) ───────────────────────────────
    if show_counterfactual:
        sample = steps_df[(steps_df["type"] == "counterfactual") & (steps_df["subject_id"].isin(bulls_shown))]
        sample = sample.iloc[::6]
        for _, r in sample.iterrows():
            folium.PolyLine(
                [[r["start_lat"], r["start_lon"]], [r["end_lat"], r["end_lon"]]],
                color="#cccccc", weight=1, opacity=0.4, dash_array="3,5",
            ).add_to(m)

    # ── Actual trajectory + forward projection ──────────────────────────────
    for bull in bulls_shown:
        grp = telemetry_df[telemetry_df["subject_id"] == bull].sort_values("hour_idx")
        coords = grp[["lat", "lon"]].values.tolist()
        colour = BULL_COLOURS.get(bull, "#52b788")
        folium.PolyLine(coords, color=colour, weight=3, opacity=0.9,
                         tooltip=f"{bull} — actual trajectory").add_to(m)
        folium.CircleMarker(coords[-1], radius=6, color="#ffffff", fill=True,
                             fill_color=colour, fill_opacity=1,
                             tooltip=f"{bull} — current position").add_to(m)

        proj = projections.get(bull)
        if proj is not None and len(proj) > 1:
            proj_coords = proj[["lat", "lon"]].values.tolist()
            folium.PolyLine(proj_coords, color="#e63946", weight=3, opacity=0.85,
                             dash_array="8,6",
                             tooltip=f"{bull} — predicted deviation path").add_to(m)
            folium.Marker(
                proj_coords[-1],
                icon=folium.Icon(color="orange", icon="exclamation-sign"),
                tooltip=f"{bull} — predicted position",
            ).add_to(m)

    # ── Fence line & community farms ────────────────────────────────────────
    folium.PolyLine(
        [[lat, lon] for lon, lat in FENCE_LINE], color="#ffd60a", weight=3, opacity=0.9,
        tooltip="Smart-Fence Line",
    ).add_to(m)
    for flon, flat, name in COMMUNITY_FARMS:
        folium.Marker(
            [flat, flon], icon=folium.Icon(color="red", icon="home"),
            tooltip=f"Community Farm — {name}",
        ).add_to(m)
    for wlon, wlat in WATER_POINTS:
        folium.CircleMarker(
            [wlat, wlon], radius=7, color="#1e90ff", fill=True, fill_color="#1e90ff",
            fill_opacity=0.8, tooltip="Water Source",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    def gradient_bar(stops: list[str]) -> str:
        return f"linear-gradient(to right, {', '.join(stops)})"

    env_legend_html = f"""
    <div style="position:fixed;bottom:30px;left:10px;z-index:1000;
                background:rgba(10,20,15,.88);padding:10px 14px;border-radius:8px;
                font-family:'Inter',sans-serif;font-size:11px;color:#ddd;line-height:1.4;width:170px">
      <div style="font-weight:800;font-size:12px;margin-bottom:6px;color:#52b788">ENVIRONMENTAL LAYERS</div>
      <div style="margin-bottom:6px">
        <div>NDVI</div>
        <div style="height:8px;border-radius:3px;background:{gradient_bar(NDVI_PALETTE)}"></div>
        <div style="display:flex;justify-content:space-between;color:#999"><span>-0.1</span><span>0.9</span></div>
      </div>
      <div style="margin-bottom:6px">
        <div>Land Surface Temp (°C)</div>
        <div style="height:8px;border-radius:3px;background:{gradient_bar(LST_PALETTE)}"></div>
        <div style="display:flex;justify-content:space-between;color:#999"><span>15</span><span>50</span></div>
      </div>
      <div>
        <div>Rainfall (mm, 10d)</div>
        <div style="height:8px;border-radius:3px;background:{gradient_bar(RAIN_PALETTE)}"></div>
        <div style="display:flex;justify-content:space-between;color:#999"><span>0</span><span>300</span></div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(env_legend_html))

    legend_html = """
    <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                background:rgba(10,20,15,.88);padding:10px 14px;border-radius:8px;
                font-family:'Inter',sans-serif;font-size:11px;color:#ddd;line-height:1.7">
      <div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#52b788">LEGEND</div>
      <div><span style="color:#2dc653">▬</span>&nbsp;Baseline Corridor</div>
      <div><span style="color:#52b788">━</span>&nbsp;Actual Path</div>
      <div><span style="color:#e63946">┄</span>&nbsp;Predicted Deviation</div>
      <div><span style="color:#cccccc">┄</span>&nbsp;Counterfactual Step</div>
      <div><span style="color:#ffd60a">━</span>&nbsp;Smart-Fence</div>
      <div><span style="color:#ff3333">▭</span>&nbsp;Bottleneck Cell</div>
      <div><span style="color:#ff0000">●</span>&nbsp;Farm &nbsp;<span style="color:#1e90ff">●</span>&nbsp;Water</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — TACTICAL EARLY-WARNING DISPATCH & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

DEVIATION_ALERT_KM = 2.5    # deviation beyond this from baseline triggers a proactive alert
FENCE_ALERT_KM     = 3.0    # proximity to fence beyond which risk is considered imminent


def build_tactical_directive(bull: str, metrics: dict, days_out: float) -> dict:
    """Compose the automated field-commander directive if thresholds are breached."""
    eta = (date.today() + timedelta(days=round(days_out))).strftime("%A %d %b")

    critical = metrics["deviation_km"] >= DEVIATION_ALERT_KM or metrics["fence_distance_km"] <= FENCE_ALERT_KM

    if not critical or days_out == 0:
        return {
            "critical": False,
            "title": f"✅ {bull} — Nominal",
            "body": (
                f"Projected deviation {metrics['deviation_km']} km from baseline corridor; "
                f"{metrics['fence_distance_km']} km from nearest fence segment. No pre-emptive "
                f"action required at this horizon."
            ),
        }

    return {
        "critical": True,
        "title": f"🚨 PROACTIVE ALERT — {bull}",
        "body": (
            f"<b>{bull}</b> predicted to deviate <b>{metrics['deviation_km']} km</b> from its "
            f"historical baseline corridor by <b>{eta}</b>, driven by acute thermal stress and "
            f"water-pan drawdown. Projected position sits <b>{metrics['fence_distance_km']} km</b> "
            f"from the smart-fence line, <b>{metrics['nearest_farm_km']} km</b> from "
            f"<b>{metrics['nearest_farm']}</b>.<br><br>"
            f"<b>Directive:</b> Dispatch a mobile ranger unit to the smart-fence segment nearest "
            f"{metrics['nearest_farm']} and notify community scouts within "
            f"<b>{metrics['nearest_farm']}</b> within the next 48 hours to pre-empt conflict "
            f"before damage occurs."
        ),
    }


def render_dispatch_panel(directive: dict) -> None:
    css_class = "alert-critical" if directive["critical"] else "alert-ok"
    st.markdown(
        f"<div class='{css_class}'>"
        f"  <div class='alert-title'>{directive['title']}</div>"
        f"  <div class='alert-body'>{directive['body']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='text-align:center;padding:10px 0 6px'>
      <div style='font-size:2rem'>🐘</div>
      <div style='font-size:1.0rem;font-weight:800;color:#52b788;letter-spacing:1px'>
        PREDICTIVE CORRIDOR DSS
      </div>
      <div style='font-size:.65rem;color:#7a9a87;margin-top:2px'>
        Laikipia–Samburu
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    gee_ready, gee_msg = init_gee()
    if gee_ready:
        st.sidebar.success(f"🛰️ {gee_msg}", icon="✅")
    else:
        st.sidebar.warning(f"🛰️ Live satellite data unavailable — using synthetic fallback.\n\n{gee_msg}", icon="⚠️")

    st.sidebar.markdown("**🕒 Temporal Mode**")
    temporal_mode = st.sidebar.radio(
        "Mode", ["📅 Historical Review", "🔴 Live (Today)", "🔮 Predictive Forecast"], index=1,
        label_visibility="collapsed",
    )

    today = date.today()
    hist_floor = today - timedelta(days=HIST_DAYS)
    timeline_label = "Today (Live)"
    days_out = 0.0
    hist_range: tuple[date, date] | None = None

    if temporal_mode == "📅 Historical Review":
        pick_mode = st.sidebar.radio("Pick", ["Single date", "Date range"], horizontal=True)
        if pick_mode == "Single date":
            d = st.sidebar.date_input(
                "Historical date", value=today - timedelta(days=7),
                min_value=hist_floor, max_value=today - timedelta(days=1),
            )
            hist_range = (d, d)
        else:
            d0, d1 = st.sidebar.date_input(
                "Historical range", value=(today - timedelta(days=14), today - timedelta(days=8)),
                min_value=hist_floor, max_value=today - timedelta(days=1),
            )
            hist_range = (min(d0, d1), max(d0, d1))
        timeline_label = f"Historical Review — {hist_range[0]:%d %b} to {hist_range[1]:%d %b}"
        days_out = 0.0
        ref_date_for_tiles = hist_range[1].isoformat()

    elif temporal_mode == "🔴 Live (Today)":
        timeline_label = "Today (Live)"
        days_out = 0.0
        ref_date_for_tiles = today.isoformat()

    else:  # Predictive Forecast
        pick_mode = st.sidebar.radio("Pick", ["Single date", "Date range"], horizontal=True)
        if pick_mode == "Single date":
            d = st.sidebar.date_input(
                "Forecast date", value=today + timedelta(days=7),
                min_value=today + timedelta(days=1), max_value=today + timedelta(days=MAX_FORECAST_DAYS),
            )
            days_out = float((d - today).days)
            timeline_label = f"Predictive Forecast — {d:%A %d %b} (+{int(days_out)}d)"
        else:
            d0, d1 = st.sidebar.date_input(
                "Forecast range", value=(today + timedelta(days=3), today + timedelta(days=10)),
                min_value=today + timedelta(days=1), max_value=today + timedelta(days=MAX_FORECAST_DAYS),
            )
            d1 = max(d0, d1)
            days_out = float((d1 - today).days)
            timeline_label = f"Predictive Forecast — {d0:%d %b} to {d1:%d %b} (+{int(days_out)}d)"
        ref_date_for_tiles = today.isoformat()  # GEE has no future imagery — latest observed reference

    timeline_params = scenario_for_days_out(days_out)

    st.sidebar.divider()
    st.sidebar.markdown("**🐘 Telemetry Display**")
    bulls_shown = st.sidebar.multiselect("Collared bulls (EarthRanger feed)", BULLS, default=BULLS)
    focus_bull = st.sidebar.selectbox("Focus bull for analytics", bulls_shown or BULLS)
    show_counterfactual = st.sidebar.toggle("Show counterfactual steps", value=True)
    grid_res = st.sidebar.slider("Risk grid resolution (cells/side)", 6, 18, 12)

    st.sidebar.divider()
    st.sidebar.markdown("**🔎 Point Inspector**")
    inspect_layer = st.sidebar.radio(
        "Layer to read on map click", ["NDVI", "LST", "Rainfall"], horizontal=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='margin-bottom:2px;color:#52b788'>🐘 Predictive Corridor Decision Support System</h2>
    <p style='color:#7a9a87;font-size:.88rem;margin-top:0;margin-bottom:10px'>
      Live environmental ingestion · Behavioral lag tracking · Step-selection inference ·
      Predictive deviation forecasting
    </p>
    """, unsafe_allow_html=True)

    # ── Layer 1: live ingestion ───────────────────────────────────────────────
    env = fetch_live_environmental_layers(gee_ready, ref_date_for_tiles)
    if env["source"] == "synthetic" and "_gee_fetch_error" in st.session_state:
        st.info(
            f"GEE query fell back to synthetic data: {st.session_state['_gee_fetch_error']}",
            icon="ℹ️",
        )

    # ── Layer 2: telemetry, lag tracker, SSF ──────────────────────────────────
    telemetry_df = mock_earthranger_stream(env)
    shock_hour = telemetry_df.attrs["shock_hour"]
    climate_daily = telemetry_df.attrs["climate_daily"]
    base_date = telemetry_df["recorded_at"].min().normalize()

    lag_df = detect_behavioral_lag(telemetry_df, shock_hour)
    steps_df = generate_steps_with_alternatives(telemetry_df)
    cov_df = extract_step_covariates(steps_df, climate_daily, base_date, WATER_POINTS)
    beta_table, mu, sigma = fit_ssf_logistic(cov_df)

    is_historical = temporal_mode == "📅 Historical Review"
    if is_historical:
        hist_telemetry_df = telemetry_df[
            (telemetry_df["recorded_date"] >= hist_range[0]) & (telemetry_df["recorded_date"] <= hist_range[1])
        ]
        if hist_telemetry_df.empty:
            hist_telemetry_df = telemetry_df

    # ── Layer 3: predictive deviation per bull (suppressed in Historical mode) ─
    baseline_corridors, projections, metrics_by_bull = {}, {}, {}
    for bull in BULLS:
        baseline_corridors[bull] = build_baseline_corridor(telemetry_df, bull)
        if is_historical:
            actual = hist_telemetry_df[hist_telemetry_df["subject_id"] == bull][["lon", "lat"]].reset_index(drop=True)
            if actual.empty:
                actual = pd.DataFrame({"lon": [baseline_corridors[bull][-1][0]], "lat": [baseline_corridors[bull][-1][1]]})
            projections[bull] = actual
            metrics_by_bull[bull] = compute_deviation_metrics(baseline_corridors[bull], actual)
        else:
            proj = project_forward_trajectory(
                telemetry_df, bull, timeline_params["days_out"],
                timeline_params["temp_bump"], timeline_params["rain_decay"], timeline_params["ndvi_shock"],
            )
            projections[bull] = proj
            metrics_by_bull[bull] = compute_deviation_metrics(baseline_corridors[bull], proj)

    # ── Risk grid ──────────────────────────────────────────────────────────────
    grid_df = generate_risk_grid(BBOX, grid_res)
    risk_df = compute_grid_risk(grid_df, timeline_params, beta_table, mu, sigma, env)

    # ── KPI strip ──────────────────────────────────────────────────────────────
    mean_lag = lag_df["lag_hours_detected"].mean()
    peak_risk = risk_df["risk_score"].max()
    n_bottlenecks = int(risk_df["bottleneck"].sum())
    focus_metrics = metrics_by_bull[focus_bull]

    kpis = [
        ("Data Source",        "Live Satellite" if env["source"] == "live" else "Synthetic"),
        ("Mode",                temporal_mode.split(" ", 1)[1] if " " in temporal_mode else temporal_mode),
        ("Mean Tmax",           f"{env['mean_lst_c']:.1f}°C"),
        ("Mean Detected Lag",   f"{mean_lag:.0f} h"),
        ("Peak Bottleneck Risk", f"{peak_risk:.1f}%"),
        ("Bottleneck Cells",    f"{n_bottlenecks}"),
        (f"{focus_bull} Deviation", f"{focus_metrics['deviation_km']} km"),
        (f"{focus_bull} → Fence", f"{focus_metrics['fence_distance_km']} km"),
    ]
    cards = "".join(
        f"<div class='kpi-card'><div class='kpi-label'>{l}</div><div class='kpi-value'>{v}</div></div>"
        for l, v in kpis
    )
    st.markdown(f"<div class='kpi-bar'>{cards}</div>", unsafe_allow_html=True)

    # ── Tactical dispatch panel ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚡ Tactical Early-Warning Dispatch</div>", unsafe_allow_html=True)
    directive_cols = st.columns(len(BULLS))
    for col, bull in zip(directive_cols, BULLS):
        with col:
            if is_historical:
                directive = {
                    "critical": False,
                    "title": f"📅 {bull} — Historical Record",
                    "body": (
                        f"Actual recorded deviation {metrics_by_bull[bull]['deviation_km']} km from "
                        f"baseline corridor during {timeline_label}; review only, no forecast applied."
                    ),
                }
            else:
                directive = build_tactical_directive(bull, metrics_by_bull[bull], days_out)
            render_dispatch_panel(directive)

    # ── Map + side analytics ───────────────────────────────────────────────────
    map_col, side_col = st.columns([3, 1])

    with map_col:
        st.markdown("<div class='section-header'>Operational Landscape & Connectivity Risk</div>",
                    unsafe_allow_html=True)
        op_map = build_operational_map(
            env, telemetry_df, steps_df, risk_df, bulls_shown,
            baseline_corridors, projections, show_counterfactual, timeline_label,
        )
        map_data = st_folium(op_map, height=560, use_container_width=True)

        clicked = map_data.get("last_clicked") if map_data else None
        if clicked:
            clat, clon = clicked["lat"], clicked["lng"]
            if env.get("source") != "live":
                st.warning("Point inspect requires a live satellite connection — data currently unavailable.")
            else:
                units = {"NDVI": "", "LST": "°C", "Rainfall": "mm"}
                value = sample_live_point(inspect_layer, clat, clon, env["ref_date"])
                if value is not None:
                    st.info(f"📍 **{inspect_layer}** at ({clat:.3f}, {clon:.3f}): **{value:.3f}{units[inspect_layer]}**")
                else:
                    st.warning(f"No {inspect_layer} value available at that point.")

    with side_col:
        st.markdown("<div class='section-header'>SSF Resistance Coefficients</div>", unsafe_allow_html=True)
        st.dataframe(
            beta_table[beta_table["covariate"] != "Intercept"][["covariate", "beta"]],
            use_container_width=True, height=130, hide_index=True,
        )
        st.markdown("<div class='section-header'>Behavioral Lag (per bull)</div>", unsafe_allow_html=True)
        st.dataframe(
            lag_df[["subject_id", "lag_hours_detected", "lag_hours_true"]],
            use_container_width=True, height=130, hide_index=True,
        )

    # ── Divergence analytics ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Ancestral Baseline vs. Climate-Stressed Reality — 7-Day Divergence</div>",
                unsafe_allow_html=True)
    seven_day = scenario_for_days_out(7)
    div_df = divergence_series(
        baseline_corridors[focus_bull], telemetry_df, focus_bull,
        seven_day["temp_bump"], seven_day["rain_decay"], seven_day["ndvi_shock"],
    )
    base = alt.Chart(div_df).encode(x=alt.X("day:O", title="Days From Today"))
    line_dev = base.mark_line(point=True, color="#e63946", strokeWidth=2.5).encode(
        y=alt.Y("deviation_km:Q", title="Deviation From Baseline (km)", axis=alt.Axis(titleColor="#e63946")),
        tooltip=["day", "deviation_km"],
    )
    line_fence = base.mark_line(point=True, color="#ffd60a", strokeWidth=2.5, strokeDash=[4, 3]).encode(
        y=alt.Y("fence_distance_km:Q", title="Distance to Fence (km)", axis=alt.Axis(titleColor="#ffd60a")),
        tooltip=["day", "fence_distance_km"],
    )
    divergence_chart = alt.layer(line_dev, line_fence).resolve_scale(y="independent").properties(height=280)
    st.altair_chart(divergence_chart, use_container_width=True)

    # ── Full risk grid table ───────────────────────────────────────────────────
    with st.expander("📋 Full Bottleneck Risk Grid", expanded=False):
        display_df = risk_df[[
            "cell_id", "centroid_lat", "centroid_lon", "dist_water_km",
            "dist_fence_km", "risk_score", "bottleneck",
        ]].rename(columns={
            "cell_id": "Cell", "centroid_lat": "Lat", "centroid_lon": "Lon",
            "dist_water_km": "Water km", "dist_fence_km": "Fence km",
            "risk_score": "Risk %", "bottleneck": "Bottleneck",
        })
        st.dataframe(
            display_df, use_container_width=True, height=320,
            column_config={
                "Risk %": st.column_config.ProgressColumn(
                    "Risk %", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )


if __name__ == "__main__":
    main()
