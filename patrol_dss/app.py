"""
Tactical Patrol Decision Support System (DSS)
==============================================
Anti-poaching and human-wildlife conflict mitigation command tool for Kenya.

All spatial objects are generated synthetically — no external file paths required.

Run:
    conservation_env\\Scripts\\streamlit.exe run patrol_dss\\app.py
"""

from __future__ import annotations

import json
import math
import random

import folium
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Polygon, mapping
from streamlit_folium import st_folium

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patrol DSS — Anti-Poaching Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  section[data-testid="stSidebar"]     { background: #0d0d1a; }
  section[data-testid="stSidebar"] *   { color: #d0d0d0 !important; }
  section[data-testid="stSidebar"] hr  { border-color: #2a2a4a; }

  .stApp { background: #0f0f1e; color: #e0e0e0; }

  .kpi-bar { display:flex; gap:8px; margin-bottom:14px; align-items:stretch; flex-wrap:wrap; }
  .kpi-card {
    flex: 0 0 auto;
    background: #1a1a2e;
    border-radius: 8px;
    padding: 7px 16px;
    border-top: 3px solid #e63946;
  }
  .kpi-label {
    font-size: .62rem; color: #888; font-weight: 700;
    text-transform: uppercase; letter-spacing: .7px; white-space: nowrap;
  }
  .kpi-value {
    font-size: 1.05rem; font-weight: 800; color: #fff; white-space: nowrap;
  }

  .dispatch-header {
    font-size: .8rem; font-weight: 700; color: #e63946;
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 4px;
  }

  div[data-testid="stExpander"] { background: #1a1a2e; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BBOX = {"min_lon": 36.3, "max_lon": 37.8, "min_lat": -0.5, "max_lat": 1.2}
N_CELLS   = 100
SEED      = 42

# YlOrRd palette (10 stops, low → high risk)
YLORD = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#4d0013",
]

DIRECTIVES = [
    ("🔴", "Deploy Ambush Team"),
    ("🚗", "Mobile Vehicle Patrol"),
    ("🚁", "Request Aerial Surveillance"),
    ("👣", "Foot Patrol — 2 Rangers"),
    ("📡", "Standby + Comms Watch"),
    ("🐕", "K9 Track & Intercept"),
    ("🔒", "Perimeter Reinforcement"),
    ("📢", "Community Scout Alert"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEXAGONAL GRID GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building spatial grid…")
def generate_hex_grid(bbox: dict, n_cells: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Programmatically generate ~n_cells pointy-top hexagonal polygons over bbox.
    Each cell carries synthetic environmental baseline features.
    """
    rng = random.Random(seed)

    lon_span = bbox["max_lon"] - bbox["min_lon"]
    lat_span = bbox["max_lat"] - bbox["min_lat"]

    # Solve for hex size such that total coverage ≈ n_cells hexagons
    # Area of regular hex = (3√3 / 2) × size²
    total_area = lon_span * lat_span
    size = math.sqrt(total_area / (n_cells * 3 * math.sqrt(3) / 2)) * 0.95

    h_spacing = math.sqrt(3) * size      # pointy-top horizontal step
    v_spacing = 1.5 * size               # vertical step

    records: list[dict] = []
    cell_id = 0

    row = 0
    while True:
        cy = bbox["min_lat"] + row * v_spacing
        if cy > bbox["max_lat"] + size:
            break
        col = 0
        while True:
            cx = (
                bbox["min_lon"]
                + col * h_spacing
                + (row % 2) * (h_spacing / 2)
            )
            if cx > bbox["max_lon"] + size:
                break

            # Only keep cells whose centre is inside bbox
            if (
                bbox["min_lon"] <= cx <= bbox["max_lon"]
                and bbox["min_lat"] <= cy <= bbox["max_lat"]
            ):
                # Pointy-top hexagon vertices
                verts = [
                    (
                        cx + size * math.cos(math.radians(60 * i - 30)),
                        cy + size * math.sin(math.radians(60 * i - 30)),
                    )
                    for i in range(6)
                ]
                poly = Polygon(verts)
                records.append(
                    {
                        "cell_id":       cell_id,
                        "centroid_lon":  round(cx, 5),
                        "centroid_lat":  round(cy, 5),
                        "geometry":      poly,
                        "geojson":       json.dumps(mapping(poly)),
                        # synthetic environmental features
                        "dist_road_km":  round(rng.uniform(0.5, 25.0), 2),
                        "dist_water_km": round(rng.uniform(0.5, 15.0), 2),
                        "hist_weight":   rng.randint(0, 10),
                    }
                )
                cell_id += 1
                if cell_id >= n_cells:
                    return pd.DataFrame(records)
            col += 1
        row += 1

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPATIO-TEMPORAL RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk(
    df: pd.DataFrame,
    moon_illum: float,
    days_since_rain: int,
    w1: float,
    w2: float,
    w3: float,
) -> pd.DataFrame:
    """
    Patrol Priority Score:
      S = (w1 × Hist) + (w2 × MoonIllum × 1/DistRoad) − (w3 × DaysSinceRain × DistWater)

    Result is min-max normalised to 0–100 (risk percentage).
    """
    out = df.copy()
    raw = (
        (w1 * out["hist_weight"])
        + (w2 * moon_illum * (1.0 / out["dist_road_km"].clip(lower=0.1)))
        - (w3 * days_since_rain * out["dist_water_km"])
    )
    r_min, r_max = raw.min(), raw.max()
    out["risk_score"] = (
        ((raw - r_min) / (r_max - r_min) * 100).round(1)
        if r_max > r_min
        else pd.Series(50.0, index=out.index)
    )
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. COLOUR UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def risk_colour(score: float) -> str:
    idx = min(int(score / 100 * len(YLORD)), len(YLORD) - 1)
    return YLORD[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_patrol_map(df: pd.DataFrame, n_teams: int) -> folium.Map:
    clat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    clon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2

    m = folium.Map(location=[clat, clon], zoom_start=9, tiles="CartoDB dark_matter")

    priority_ids = set(df.head(n_teams)["cell_id"].tolist())
    priority_rank = {row["cell_id"]: i + 1 for i, row in df.head(n_teams).iterrows()}

    for _, row in df.iterrows():
        geo    = json.loads(row["geojson"])
        latlon = [[pt[1], pt[0]] for pt in geo["coordinates"][0]]

        is_pri  = row["cell_id"] in priority_ids
        fill    = risk_colour(row["risk_score"])
        border  = "#ff4444" if is_pri else "#333333"
        bweight = 2.5 if is_pri else 0.4

        rank_txt = f"<br><b>⚠️ PRIORITY #{priority_rank[row['cell_id']]}</b>" if is_pri else ""
        tooltip = (
            f"<b>Cell {row['cell_id']}</b>{rank_txt}<br>"
            f"Risk: <b>{row['risk_score']:.1f}%</b><br>"
            f"Hist Weight: {row['hist_weight']} &nbsp;|&nbsp; "
            f"Road: {row['dist_road_km']} km &nbsp;|&nbsp; "
            f"Water: {row['dist_water_km']} km"
        )

        folium.Polygon(
            locations=latlon,
            color=border,
            weight=bweight,
            fill=True,
            fill_color=fill,
            fill_opacity=0.65,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    # In-map legend
    legend_html = """
    <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                background:rgba(10,10,20,.85);padding:10px 14px;
                border-radius:8px;font-family:'Inter',sans-serif;
                font-size:11px;color:#ddd;line-height:1.8">
      <div style="font-weight:800;font-size:12px;margin-bottom:4px;
                  color:#e63946;letter-spacing:.5px">RISK LEVEL</div>
      <div><span style="color:#4d0013;font-size:16px">■</span> &nbsp;Critical</div>
      <div><span style="color:#bd0026;font-size:16px">■</span> &nbsp;Very High</div>
      <div><span style="color:#fc4e2a;font-size:16px">■</span> &nbsp;High</div>
      <div><span style="color:#feb24c;font-size:16px">■</span> &nbsp;Moderate</div>
      <div><span style="color:#ffffcc;font-size:16px;background:#333">■</span> &nbsp;Low</div>
      <div style="margin-top:6px">
        <span style="color:#ff4444;font-size:14px">━</span>
        &nbsp;Priority Deploy Zone
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# 5. DISPATCH TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_dispatch_table(df: pd.DataFrame, n_teams: int) -> str:
    top = df.head(min(5, n_teams))
    rows = []
    for i, (_, row) in enumerate(top.iterrows()):
        icon, directive = DIRECTIVES[i % len(DIRECTIVES)]
        rows.append(
            f"| **#{i+1}** | {row['centroid_lat']:.4f},&nbsp;{row['centroid_lon']:.4f} "
            f"| **{row['risk_score']:.1f}%** | {icon} {directive} |"
        )
    header = (
        "| Rank | Centroid (Lat, Lon) | Risk % | Tactical Directive |\n"
        "|:----:|:------------------:|:------:|:------------------|\n"
    )
    return header + "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='text-align:center;padding:10px 0 6px'>
      <div style='font-size:2rem'>🛡️</div>
      <div style='font-size:1.05rem;font-weight:800;color:#e63946;letter-spacing:1.5px'>
        PATROL DSS
      </div>
      <div style='font-size:.65rem;color:#777;margin-top:2px'>
        Anti-Poaching Command · Kenya
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    st.sidebar.markdown("**🌙 Environmental Conditions**")
    moon_illum = st.sidebar.slider(
        "Moon Illumination %", 0, 100, 60, 5,
        help="Higher illumination raises night-visibility poaching risk.",
    )
    days_since_rain = st.sidebar.slider(
        "Days Since Last Rain", 0, 60, 14,
        help="Dry spells push elephants toward water pans, heightening conflict.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**🚨 Deployment**")
    n_teams = st.sidebar.selectbox(
        "Available Ranger Teams", [1, 2, 3, 4, 5, 6, 8, 10], index=2,
        help="Top N high-risk cells will be flagged for deployment.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**⚖️ Risk Weight Tuning**")
    w1 = st.sidebar.slider("w₁ — Historical Incident",    0.0, 1.0, 0.40, 0.05)
    w2 = st.sidebar.slider("w₂ — Moon × Road Proximity",  0.0, 1.0, 0.35, 0.05)
    w3 = st.sidebar.slider("w₃ — Rain × Water Distance",  0.0, 1.0, 0.25, 0.05)

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-size:.65rem;color:#555;text-align:center'>"
        "S = w₁·Hist + w₂·Moon·(1/Road) − w₃·Rain·Water"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='margin-bottom:2px;color:#e63946;font-family:Inter,sans-serif'>
      🛡️ Tactical Patrol Decision Support System
    </h2>
    <p style='color:#666;font-size:.88rem;margin-top:0;margin-bottom:12px'>
      Spatio-temporal poaching risk modelling &amp; ranger dispatch optimisation · Kenya
    </p>
    """, unsafe_allow_html=True)

    # ── Compute ───────────────────────────────────────────────────────────────
    grid_df = generate_hex_grid(BBOX, N_CELLS, SEED)
    risk_df = compute_risk(grid_df, moon_illum / 100.0, days_since_rain, w1, w2, w3)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    top_risk     = risk_df.iloc[0]["risk_score"]
    mean_risk    = risk_df["risk_score"].mean()
    critical_n   = int((risk_df["risk_score"] >= 75).sum())
    high_n       = int((risk_df["risk_score"] >= 50).sum())

    kpis = [
        ("Grid Cells",        f"{N_CELLS}"),
        ("Peak Risk",         f"{top_risk:.1f}%"),
        ("Mean Risk",         f"{mean_risk:.1f}%"),
        ("Critical Cells",    f"{critical_n}"),
        ("High-Risk Cells",   f"{high_n}"),
        ("Teams Deploying",   f"{n_teams}"),
        ("Moon",              f"{moon_illum}%"),
        ("Dry Days",          f"{days_since_rain}d"),
    ]
    cards = "".join(
        f"<div class='kpi-card'>"
        f"  <div class='kpi-label'>{lbl}</div>"
        f"  <div class='kpi-value'>{val}</div>"
        f"</div>"
        for lbl, val in kpis
    )
    st.markdown(f"<div class='kpi-bar'>{cards}</div>", unsafe_allow_html=True)

    # ── Main layout ───────────────────────────────────────────────────────────
    map_col, right_col = st.columns([3, 1])

    with map_col:
        patrol_map = build_patrol_map(risk_df, n_teams)
        st_folium(patrol_map, height=560, use_container_width=True)

    with right_col:
        st.markdown("<div class='dispatch-header'>⚡ Dispatch Orders</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"*Moon {moon_illum}% · {days_since_rain}d dry · {n_teams} team(s)*"
        )
        st.markdown(build_dispatch_table(risk_df, n_teams))

        st.divider()
        st.markdown("<div class='dispatch-header'>📊 Risk Distribution</div>",
                    unsafe_allow_html=True)
        bins   = [0, 25, 50, 70, 85, 100]
        labels = ["Low", "Moderate", "High", "Very High", "Critical"]
        hist_data = (
            pd.cut(risk_df["risk_score"], bins=bins, labels=labels)
            .value_counts()
            .sort_index()
        )
        st.bar_chart(hist_data, height=180)

    # ── Full grid table ───────────────────────────────────────────────────────
    with st.expander("📋 Full Grid Risk Table", expanded=False):
        display_df = (
            risk_df[[
                "cell_id", "centroid_lat", "centroid_lon",
                "hist_weight", "dist_road_km", "dist_water_km", "risk_score",
            ]]
            .rename(columns={
                "cell_id":       "Cell",
                "centroid_lat":  "Lat",
                "centroid_lon":  "Lon",
                "hist_weight":   "Hist Wt",
                "dist_road_km":  "Road km",
                "dist_water_km": "Water km",
                "risk_score":    "Risk %",
            })
        )
        st.dataframe(
            display_df.style.background_gradient(subset=["Risk %"], cmap="YlOrRd"),
            use_container_width=True,
            height=320,
        )


if __name__ == "__main__":
    main()
