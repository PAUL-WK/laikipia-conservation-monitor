"""
Adaptive Spatial-Temporal Responses of African Elephants to Climatic Volatility
================================================================================
MSc Thesis Decision Support Prototype — Predictive Framework for Landscape
Connectivity, Laikipia Ecosystem, Kenya.

Three integrated research objectives, all on synthetic (internally generated)
data so the app is fully self-contained:

  A. Climatic Whiplash → Behavioral Lag Engine
  B. Step-Selection Function (SSF) discrete-choice pipeline
  C. Predictive HEC risk & connectivity corridor simulation (2030 / 2050)

Run:
    conservation_env\\Scripts\\streamlit.exe run elephant_thesis_dss\\app.py
"""

from __future__ import annotations

import math
from datetime import timedelta

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Elephant Climatic Volatility DSS",
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

  div[data-testid="stExpander"] { background:#16261d; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BBOX = {"min_lon": 36.85, "max_lon": 37.30, "min_lat": 0.15, "max_lat": 0.55}
BULLS = ["Bull_M1", "Bull_M2", "Bull_M3"]
BULL_COLOURS = {"Bull_M1": "#52b788", "Bull_M2": "#4895ef", "Bull_M3": "#f3722c"}
N_DAYS = 21          # captures a dry-spell → storm-shock → recovery cycle
SEED = 42

WATER_POINTS = [(36.95, 0.22), (37.20, 0.48), (37.05, 0.35)]   # permanent water (lon, lat)
INFRA_POINT  = (37.22, 0.20)                                    # farm / settlement cluster

SCENARIOS = {
    "Current Baseline":          {"temp_offset": 0.0, "water_availability": 1.00, "ndvi_volatility": 1.00, "decline_k": 0.004},
    "2030 IPCC Climate Shock":   {"temp_offset": 2.0, "water_availability": 0.70, "ndvi_volatility": 1.40, "decline_k": 0.018},
    "2050 Intense Volatility":   {"temp_offset": 4.0, "water_availability": 0.45, "ndvi_volatility": 2.00, "decline_k": 0.035},
}

RISK_PALETTE = [
    "#ffffb2", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
]


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE A — CLIMATIC WHIPLASH → BEHAVIORAL LAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_climate_series(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """
    Daily Tmax (°C), cumulative 3-day rainfall (mm), and 8-day NDVI anomaly.
    Simulates an escalating dry spell (days 0-13), a sudden intense storm
    (days 14-15), then partial vegetation recovery.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-02-01", periods=n_days, freq="D")

    tmax = np.zeros(n_days)
    rain_daily = np.zeros(n_days)
    for i in range(n_days):
        if i < 14:
            tmax[i] = 30 + i * 0.68 + rng.normal(0, 0.4)
            rain_daily[i] = max(0.0, rng.normal(0.2, 0.3))
        elif i < 16:
            tmax[i] = 24 + rng.normal(0, 0.5)
            rain_daily[i] = rng.uniform(35, 55)
        else:
            tmax[i] = 26 + rng.normal(0, 1.0)
            rain_daily[i] = max(0.0, rng.normal(3.0, 2.0))

    rain_3d = pd.Series(rain_daily).rolling(3, min_periods=1).sum().values

    ndvi = np.zeros(n_days)
    val = 0.0
    for i in range(n_days):
        val += (-0.035 if i < 14 else 0.045) + rng.normal(0, 0.01)
        ndvi[i] = np.clip(val, -1.0, 1.0)

    return pd.DataFrame({
        "date":         dates,
        "tmax_c":       tmax.round(1),
        "rain_3d_mm":   rain_3d.round(1),
        "ndvi_anomaly": ndvi.round(3),
    })


def detect_shock_day(climate_df: pd.DataFrame, temp_thresh: float = 38.0,
                      ndvi_thresh: float = -0.4) -> int:
    """First day index where Tmax > thresh AND NDVI anomaly < thresh (the climatic shock)."""
    hit = climate_df[(climate_df["tmax_c"] > temp_thresh) & (climate_df["ndvi_anomaly"] < ndvi_thresh)]
    return int(hit.index[0]) if not hit.empty else int(climate_df["tmax_c"].idxmax())


@st.cache_data(show_spinner="Simulating hourly telemetry…")
def generate_telemetry(climate_df: pd.DataFrame, shock_day: int, seed: int = SEED) -> pd.DataFrame:
    """
    Hourly GPS telemetry for 3 collared bulls. Each bull carries a hidden
    ground-truth behavioral-lag (hours after the shock before movement
    visibly shifts toward human infrastructure) — used later to validate
    the statistical detector in `detect_behavioral_lag`.
    """
    rng = np.random.default_rng(seed)
    n_hours = len(climate_df) * 24
    hourly_idx = pd.date_range(climate_df["date"].min(), periods=n_hours, freq="h")
    day_of_hour = np.repeat(np.arange(len(climate_df)), 24)[:n_hours]
    shock_hour = shock_day * 24

    records = []
    for bull in BULLS:
        home_lon = 37.0 + rng.uniform(-0.08, 0.08)
        home_lat = 0.35 + rng.uniform(-0.10, 0.10)
        lon, lat = home_lon, home_lat
        bearing = rng.uniform(0, 360)
        lag_hours_true = int(rng.integers(6, 30))           # hidden ground truth
        response_hour = shock_hour + lag_hours_true

        for h in range(n_hours):
            d = day_of_hour[h]
            tmax = climate_df.loc[d, "tmax_c"]
            ndvi = climate_df.loc[d, "ndvi_anomaly"]

            heat_stress = max(0.0, tmax - 32) / 10.0
            base_step = 0.0025 + heat_stress * 0.0018
            bearing = (bearing + rng.normal(0, 25)) % 360

            if h >= response_hour:
                dlon, dlat = INFRA_POINT[0] - lon, INFRA_POINT[1] - lat
                target_bearing = math.degrees(math.atan2(dlon, dlat)) % 360
                bearing = (0.55 * bearing + 0.45 * target_bearing) % 360
                step = base_step * 2.3
            else:
                step = base_step

            rad = math.radians(bearing)
            lon += step * math.sin(rad)
            lat += step * math.cos(rad)

            records.append({
                "animal_id": bull, "timestamp": hourly_idx[h], "hour_idx": h,
                "lon": round(lon, 6), "lat": round(lat, 6),
                "tmax_c": tmax, "ndvi_anomaly": ndvi,
                "lag_hours_true": lag_hours_true, "response_hour_idx": response_hour,
            })

    return pd.DataFrame(records)


def detect_behavioral_lag(telemetry_df: pd.DataFrame, shock_hour: int) -> pd.DataFrame:
    """
    Statistically detect the Behavioral Lag Time: hours between the climatic
    shock and the first sustained (3h rolling mean) breach of a 1.5σ threshold
    on the *rate of approach toward human infrastructure* — this isolates the
    directional behavioral response from heat-driven baseline restlessness
    (which drifts gradually and would otherwise contaminate a raw-velocity test).
    """
    rows = []
    for bull, grp in telemetry_df.groupby("animal_id"):
        grp = grp.sort_values("hour_idx").reset_index(drop=True)
        grp["dist_infra_km"] = grp.apply(
            lambda r: haversine_km(r["lon"], r["lat"], *INFRA_POINT), axis=1
        )
        # approach rate = decrease in distance to infrastructure per hour (positive = approaching)
        grp["approach_rate"] = -grp["dist_infra_km"].diff().fillna(0.0)

        baseline = grp.loc[grp["hour_idx"] < shock_hour, "approach_rate"]
        mu, sigma = baseline.mean(), baseline.std(ddof=0) or 1e-6
        threshold = mu + 2.5 * sigma

        # require a sustained (6h rolling mean) breach to reject transient noise
        post = grp.loc[grp["hour_idx"] >= shock_hour].copy()
        post["roll"] = post["approach_rate"].rolling(6, min_periods=6).mean()
        exceed = post.loc[post["roll"] > threshold]

        lag_detected = int(exceed.iloc[0]["hour_idx"] - shock_hour) if not exceed.empty else np.nan
        lag_true = int(grp["lag_hours_true"].iloc[0])

        rows.append({
            "animal_id": bull, "lag_hours_detected": lag_detected,
            "lag_hours_true": lag_true, "baseline_approach_km_h": round(mu, 4),
            "approach_threshold_km_h": round(threshold, 4),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE B — STEP-SELECTION FUNCTION (SSF) PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating counterfactual steps…")
def generate_steps_with_alternatives(telemetry_df: pd.DataFrame, n_alt: int = 4,
                                      seed: int = SEED) -> pd.DataFrame:
    """
    For every actual step, draw n_alt counterfactual steps by resampling the
    empirical step-length distribution and a Gaussian turn-angle distribution
    around the previous bearing — the standard SSF data-structuring approach.
    """
    rng = np.random.default_rng(seed)
    out = []

    for bull, grp in telemetry_df.groupby("animal_id"):
        grp = grp.sort_values("hour_idx").reset_index(drop=True)
        # subsample every 4th hourly fix → manageable "steps" for the SSF design
        grp = grp.iloc[::4].reset_index(drop=True)
        lons, lats, ts = grp["lon"].values, grp["lat"].values, grp["timestamp"].values

        emp_steps, emp_bearings = [], []
        for i in range(1, len(grp)):
            dlon, dlat = lons[i] - lons[i - 1], lats[i] - lats[i - 1]
            emp_steps.append(math.hypot(dlon, dlat))
            emp_bearings.append(math.degrees(math.atan2(dlon, dlat)) % 360)
        emp_steps = np.array(emp_steps) if emp_steps else np.array([0.001])

        for i in range(1, len(grp)):
            sl, sla = lons[i - 1], lats[i - 1]
            out.append({
                "animal_id": bull, "step_id": i, "type": "actual", "label": 1,
                "start_lon": sl, "start_lat": sla,
                "end_lon": lons[i], "end_lat": lats[i], "timestamp": ts[i],
            })
            prev_bearing = emp_bearings[i - 1]
            for _ in range(n_alt):
                rand_len = rng.choice(emp_steps)
                cf_bearing = (prev_bearing + rng.normal(0, 45)) % 360
                rad = math.radians(cf_bearing)
                out.append({
                    "animal_id": bull, "step_id": i, "type": "counterfactual", "label": 0,
                    "start_lon": sl, "start_lat": sla,
                    "end_lon": sl + rand_len * math.sin(rad),
                    "end_lat": sla + rand_len * math.cos(rad),
                    "timestamp": ts[i],
                })
    return pd.DataFrame(out)


def extract_step_covariates(steps_df: pd.DataFrame, climate_df: pd.DataFrame,
                             water_points: list[tuple[float, float]]) -> pd.DataFrame:
    """Attach Climate + Biophysical covariates to every actual/counterfactual endpoint."""
    df = steps_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("D")
    clim = climate_df.set_index("date")

    df["thermal_stress"] = df["date"].map(clim["tmax_c"]).astype(float)
    df["veg_density"] = df["date"].map(clim["ndvi_anomaly"]).apply(lambda v: (v + 1) / 2).astype(float)
    df["dist_water_km"] = df.apply(
        lambda r: min(haversine_km(r["end_lon"], r["end_lat"], wlon, wlat) for wlon, wlat in water_points),
        axis=1,
    )
    return df


def fit_ssf_logistic(df: pd.DataFrame, features=("thermal_stress", "veg_density", "dist_water_km"),
                      lr: float = 0.15, n_iter: int = 800) -> pd.DataFrame:
    """
    Pooled discrete-choice logistic regression (gradient descent, numpy-only)
    approximating a conditional SSF: P(actual selection) ~ covariates.
    Returns standardised beta coefficients, sign indicates selection (+) vs
    avoidance (-) under current climate volatility.
    """
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
        lambda b: "Selecting (avoids climate stress)" if b > 0 else "Avoiding / stress-driven deviation"
    )
    return table, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE C — PREDICTIVE HEC RISK & CONNECTIVITY CORRIDOR SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building risk grid…")
def generate_risk_grid(bbox: dict, n_per_side: int = 12, seed: int = SEED) -> pd.DataFrame:
    """Square spatial matrix over the bbox carrying baseline covariates."""
    rng = np.random.default_rng(seed)
    lons = np.linspace(bbox["min_lon"], bbox["max_lon"], n_per_side + 1)
    lats = np.linspace(bbox["min_lat"], bbox["max_lat"], n_per_side + 1)

    records = []
    cell_id = 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            lon0, lon1 = lons[i], lons[i + 1]
            lat0, lat1 = lats[j], lats[j + 1]
            clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
            dist_water = min(haversine_km(clon, clat, wlon, wlat) for wlon, wlat in WATER_POINTS)
            dist_infra = haversine_km(clon, clat, *INFRA_POINT)
            records.append({
                "cell_id": cell_id, "lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
                "centroid_lon": clon, "centroid_lat": clat,
                "dist_water_km": dist_water, "dist_infra_km": dist_infra,
                "base_ndvi": rng.uniform(-0.1, 0.3),
            })
            cell_id += 1
    return pd.DataFrame(records)


def apply_scenario(grid_df: pd.DataFrame, climate_df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """Project grid covariates forward under a climate scenario."""
    df = grid_df.copy()
    baseline_tmax = climate_df["tmax_c"].max()
    df["thermal_stress"] = baseline_tmax + scenario["temp_offset"]
    df["dist_water_km"] = df["dist_water_km"] / max(scenario["water_availability"], 0.05)
    df["veg_density"] = ((df["base_ndvi"] - 0.25 * scenario["ndvi_volatility"]).clip(-1, 1) + 1) / 2
    return df


def compute_hec_risk(grid_df: pd.DataFrame, beta_table: pd.DataFrame, mu, sigma) -> pd.DataFrame:
    """
    Combine the SSF selection probability (animal pulled toward stressed
    cells) with proximity to human infrastructure to yield a HEC probability.
    """
    df = grid_df.copy()
    feats = ["thermal_stress", "veg_density", "dist_water_km"]
    X = df[feats].to_numpy(dtype=float)
    Xs = (X - mu) / sigma
    beta = beta_table.set_index("covariate")["beta"]
    z = beta["Intercept"] + Xs @ beta[feats].to_numpy()
    selection_p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    proximity_w = 1.0 / (1.0 + df["dist_infra_km"] / 5.0)
    raw = selection_p * proximity_w
    df["hec_probability"] = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 100).round(1)
    df["critical_bottleneck"] = df["hec_probability"] >= df["hec_probability"].quantile(0.80)
    return df.sort_values("hec_probability", ascending=False).reset_index(drop=True)


def risk_colour(score: float) -> str:
    idx = min(int(score / 100 * len(RISK_PALETTE)), len(RISK_PALETTE) - 1)
    return RISK_PALETTE[idx]


def corridor_viability_series() -> pd.DataFrame:
    """Synthetic 2026-2050 corridor connectivity attrition per scenario."""
    years = list(range(2026, 2051))
    rows = []
    for name, params in SCENARIOS.items():
        k = params["decline_k"]
        for y in years:
            viability = 100 * math.exp(-k * (y - 2026))
            rows.append({"year": y, "scenario": name, "viability_pct": round(viability, 1)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_corridor_map(telemetry_df: pd.DataFrame, steps_df: pd.DataFrame,
                        risk_df: pd.DataFrame, bulls_shown: list[str],
                        show_counterfactual: bool) -> folium.Map:
    clat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    clon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2
    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB dark_matter")

    # risk grid polygons
    for _, row in risk_df.iterrows():
        bounds = [[row["lat0"], row["lon0"]], [row["lat1"], row["lon1"]]]
        is_crit = bool(row["critical_bottleneck"])
        folium.Rectangle(
            bounds=bounds,
            color="#ff4444" if is_crit else "#444444",
            weight=2.2 if is_crit else 0.3,
            fill=True,
            fill_color=risk_colour(row["hec_probability"]),
            fill_opacity=0.55,
            tooltip=(
                f"Cell {row['cell_id']}<br>HEC Risk: <b>{row['hec_probability']:.1f}%</b>"
                + ("<br><b>⚠️ Critical Bottleneck</b>" if is_crit else "")
            ),
        ).add_to(m)

    # counterfactual (dashed, sparse subset for clarity)
    if show_counterfactual:
        sample = steps_df[(steps_df["type"] == "counterfactual") & (steps_df["animal_id"].isin(bulls_shown))]
        sample = sample.iloc[::6]   # thin out for readability
        for _, r in sample.iterrows():
            folium.PolyLine(
                [[r["start_lat"], r["start_lon"]], [r["end_lat"], r["end_lon"]]],
                color="#cccccc", weight=1, opacity=0.45, dash_array="4,5",
            ).add_to(m)

    # actual elephant trajectories
    for bull in bulls_shown:
        grp = telemetry_df[telemetry_df["animal_id"] == bull].sort_values("hour_idx")
        coords = grp[["lat", "lon"]].values.tolist()
        folium.PolyLine(
            coords, color=BULL_COLOURS.get(bull, "#52b788"), weight=3, opacity=0.9,
            tooltip=f"{bull} — actual trajectory",
        ).add_to(m)
        folium.CircleMarker(
            coords[-1], radius=6, color="#ffffff",
            fill=True, fill_color=BULL_COLOURS.get(bull, "#52b788"), fill_opacity=1,
            tooltip=f"{bull} — current position",
        ).add_to(m)

    # infrastructure & water markers
    folium.Marker(
        [INFRA_POINT[1], INFRA_POINT[0]],
        icon=folium.Icon(color="red", icon="home"),
        tooltip="Human Infrastructure / Settlement",
    ).add_to(m)
    for wlon, wlat in WATER_POINTS:
        folium.CircleMarker(
            [wlat, wlon], radius=7, color="#1e90ff", fill=True, fill_color="#1e90ff",
            fill_opacity=0.8, tooltip="Permanent Water Source",
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                background:rgba(10,20,15,.85);padding:10px 14px;border-radius:8px;
                font-family:'Inter',sans-serif;font-size:11px;color:#ddd;line-height:1.7">
      <div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#52b788">HEC RISK</div>
      <div><span style="color:#800026;font-size:16px">■</span>&nbsp;Critical</div>
      <div><span style="color:#fc4e2a;font-size:16px">■</span>&nbsp;High</div>
      <div><span style="color:#feb24c;font-size:16px">■</span>&nbsp;Moderate</div>
      <div><span style="color:#ffffb2;font-size:16px;background:#333">■</span>&nbsp;Low</div>
      <div style="margin-top:6px"><span style="color:#ff4444">▭</span>&nbsp;Critical Bottleneck</div>
      <div><span style="color:#cccccc">┄</span>&nbsp;Counterfactual Step</div>
      <div><span style="color:#ff0000">●</span>&nbsp;Settlement &nbsp;
           <span style="color:#1e90ff">●</span>&nbsp;Water</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='text-align:center;padding:10px 0 6px'>
      <div style='font-size:2rem'>🐘</div>
      <div style='font-size:1.0rem;font-weight:800;color:#52b788;letter-spacing:1px'>
        CLIMATIC VOLATILITY DSS
      </div>
      <div style='font-size:.65rem;color:#7a9a87;margin-top:2px'>
        Laikipia Connectivity Framework
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    st.sidebar.markdown("**🌍 Climate Scenario**")
    scenario_name = st.sidebar.selectbox("Projection horizon", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]

    st.sidebar.divider()
    st.sidebar.markdown("**🐘 Telemetry Display**")
    bulls_shown = st.sidebar.multiselect("Collared bulls", BULLS, default=BULLS)
    show_counterfactual = st.sidebar.toggle("Show counterfactual steps", value=True)
    grid_res = st.sidebar.slider("Risk grid resolution (cells/side)", 6, 18, 12)

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-size:.62rem;color:#5f7d6c;text-align:center'>"
        "Shock: Tmax&gt;38°C ∧ NDVI&lt;-0.4<br>"
        "Lag: hours to 1.5σ velocity breach"
        "</div>", unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='margin-bottom:2px;color:#52b788'>🐘 Climatic Volatility & Connectivity DSS</h2>
    <p style='color:#7a9a87;font-size:.88rem;margin-top:0;margin-bottom:10px'>
      Behavioral lag detection · Step-selection inference · Predictive HEC corridor risk
    </p>
    """, unsafe_allow_html=True)

    # ── Pipeline execution ────────────────────────────────────────────────────
    climate_df = generate_climate_series()
    shock_day = detect_shock_day(climate_df)
    telemetry_df = generate_telemetry(climate_df, shock_day)
    lag_df = detect_behavioral_lag(telemetry_df, shock_day * 24)

    steps_df = generate_steps_with_alternatives(telemetry_df)
    cov_df = extract_step_covariates(steps_df, climate_df, WATER_POINTS)
    beta_table, mu, sigma = fit_ssf_logistic(cov_df)

    risk_grid = generate_risk_grid(BBOX, grid_res)
    scenario_grid = apply_scenario(risk_grid, climate_df, scenario)
    hec_df = compute_hec_risk(scenario_grid, beta_table, mu, sigma)

    viability_df = corridor_viability_series()
    current_viability = viability_df[
        (viability_df["scenario"] == scenario_name) & (viability_df["year"] == 2026)
    ]["viability_pct"].iloc[0]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    mean_lag = lag_df["lag_hours_detected"].mean()
    peak_risk = hec_df["hec_probability"].max()
    n_critical = int(hec_df["critical_bottleneck"].sum())

    kpis = [
        ("Scenario",            scenario_name.split()[0]),
        ("Shock Day",           f"Day {shock_day}"),
        ("Mean Detected Lag",   f"{mean_lag:.0f} h"),
        ("Peak HEC Risk",       f"{peak_risk:.1f}%"),
        ("Critical Bottlenecks", f"{n_critical}"),
        ("Water Availability",  f"{scenario['water_availability']*100:.0f}%"),
        ("Corridor Viability",  f"{current_viability:.1f}%"),
    ]
    cards = "".join(
        f"<div class='kpi-card'><div class='kpi-label'>{l}</div><div class='kpi-value'>{v}</div></div>"
        for l, v in kpis
    )
    st.markdown(f"<div class='kpi-bar'>{cards}</div>", unsafe_allow_html=True)

    # ── Map + side analytics ──────────────────────────────────────────────────
    map_col, side_col = st.columns([3, 1])

    with map_col:
        st.markdown("<div class='section-header'>Connectivity Corridor & HEC Risk Map</div>",
                    unsafe_allow_html=True)
        corridor_map = build_corridor_map(telemetry_df, steps_df, hec_df, bulls_shown, show_counterfactual)
        st_folium(corridor_map, height=560, use_container_width=True)

    with side_col:
        st.markdown("<div class='section-header'>SSF Selection Coefficients</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            beta_table[beta_table["covariate"] != "Intercept"][["covariate", "beta"]],
            use_container_width=True, height=140, hide_index=True,
        )
        st.markdown("<div class='section-header'>Behavioral Lag (per bull)</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            lag_df[["animal_id", "lag_hours_detected", "lag_hours_true"]],
            use_container_width=True, height=140, hide_index=True,
        )

    # ── Altair analytics ──────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<div class='section-header'>Time-Lag Distribution</div>", unsafe_allow_html=True)
        lag_chart = (
            alt.Chart(lag_df.dropna(subset=["lag_hours_detected"]))
            .mark_bar(color="#52b788")
            .encode(
                x=alt.X("animal_id:N", title="Collared Bull"),
                y=alt.Y("lag_hours_detected:Q", title="Detected Lag (hours)"),
                tooltip=["animal_id", "lag_hours_detected", "lag_hours_true"],
            )
            .properties(height=260)
        )
        st.altair_chart(lag_chart, use_container_width=True)

    with chart_col2:
        st.markdown("<div class='section-header'>Corridor Viability Attrition 2026–2050</div>",
                    unsafe_allow_html=True)
        viability_chart = (
            alt.Chart(viability_df)
            .mark_line(point=False, strokeWidth=2.5)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("viability_pct:Q", title="Corridor Viability (%)"),
                color=alt.Color("scenario:N", title="Scenario",
                                 scale=alt.Scale(range=["#52b788", "#f3a712", "#e63946"])),
                tooltip=["scenario", "year", "viability_pct"],
            )
            .properties(height=260)
        )
        st.altair_chart(viability_chart, use_container_width=True)

    # ── Full risk grid table ──────────────────────────────────────────────────
    with st.expander("📋 Full HEC Risk Grid", expanded=False):
        display_df = hec_df[[
            "cell_id", "centroid_lat", "centroid_lon", "dist_water_km",
            "dist_infra_km", "thermal_stress", "veg_density", "hec_probability", "critical_bottleneck",
        ]].rename(columns={
            "cell_id": "Cell", "centroid_lat": "Lat", "centroid_lon": "Lon",
            "dist_water_km": "Water km", "dist_infra_km": "Infra km",
            "thermal_stress": "Tmax °C", "veg_density": "Veg Density",
            "hec_probability": "HEC Risk %", "critical_bottleneck": "Critical",
        })
        st.dataframe(
            display_df, use_container_width=True, height=320,
            column_config={
                "HEC Risk %": st.column_config.ProgressColumn(
                    "HEC Risk %", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )


if __name__ == "__main__":
    main()
