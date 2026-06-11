import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="HEC Prediction Dashboard — Space for Giants",
    page_icon="🐘",
    layout="wide",
)

# ── Synthetic Dataset ──────────────────────────────────────────────────────────

@st.cache_data
def generate_fence_data():
    np.random.seed(42)
    segments = [
        {"id": "FS-01", "name": "Ol Pejeta North",      "lat": 0.020, "lon": 36.920, "length_km": 4.2},
        {"id": "FS-02", "name": "Ol Pejeta East",       "lat": 0.005, "lon": 36.975, "length_km": 3.8},
        {"id": "FS-03", "name": "Lewa West Corridor",   "lat": 0.190, "lon": 37.390, "length_km": 6.1},
        {"id": "FS-04", "name": "Lewa South Boundary",  "lat": 0.150, "lon": 37.440, "length_km": 5.3},
        {"id": "FS-05", "name": "Borana Ranch North",   "lat": 0.490, "lon": 37.090, "length_km": 7.2},
        {"id": "FS-06", "name": "Borana Ranch South",   "lat": 0.420, "lon": 37.120, "length_km": 4.9},
        {"id": "FS-07", "name": "Mpala Research Gate",  "lat": 0.290, "lon": 36.890, "length_km": 2.8},
        {"id": "FS-08", "name": "Sosian West",          "lat": 0.550, "lon": 36.750, "length_km": 5.5},
        {"id": "FS-09", "name": "Il Ngwesi Community",  "lat": 0.620, "lon": 37.350, "length_km": 8.0},
        {"id": "FS-10", "name": "Segera Retreat",       "lat": 0.370, "lon": 36.980, "length_km": 3.3},
        {"id": "FS-11", "name": "Mugie Conservancy",    "lat": 0.710, "lon": 36.560, "length_km": 9.1},
        {"id": "FS-12", "name": "Ol Jogi South",        "lat": 0.460, "lon": 37.250, "length_km": 6.7},
        {"id": "FS-13", "name": "Colcheccio Border",    "lat": 0.100, "lon": 37.200, "length_km": 4.4},
        {"id": "FS-14", "name": "Kisima Farm Gate",     "lat": 0.580, "lon": 37.480, "length_km": 3.1},
        {"id": "FS-15", "name": "Nanyuki Periphery",    "lat": 0.015, "lon": 37.070, "length_km": 5.8},
    ]
    records = []
    for s in segments:
        historic_breaches  = int(np.random.poisson(lam=12))
        dist_water_km      = round(np.random.uniform(0.5, 8.0), 2)
        ndvi_anomaly       = round(np.random.uniform(-0.45, 0.30), 3)
        elephant_activity  = round(np.random.uniform(1.0, 10.0), 1)
        fence_condition    = np.random.choice(["Good", "Fair", "Poor"], p=[0.4, 0.35, 0.25])
        records.append({
            **s,
            "historic_breaches":  historic_breaches,
            "dist_water_km":      dist_water_km,
            "ndvi_anomaly":       ndvi_anomaly,
            "elephant_activity":  elephant_activity,
            "fence_condition":    fence_condition,
        })
    return pd.DataFrame(records)


@st.cache_data
def generate_breach_history():
    np.random.seed(7)
    hours  = np.random.choice(range(24), size=320, p=_hour_weights())
    months = np.random.choice(range(1, 13), size=320,
                              p=[0.06,0.05,0.07,0.08,0.10,0.11,0.12,0.11,0.09,0.08,0.07,0.06])
    drivers = np.random.choice(
        ["Low Rainfall", "Full Moon", "High Elephant Activity", "Fence Damage", "Crop Season"],
        size=320, p=[0.30, 0.22, 0.25, 0.13, 0.10]
    )
    return pd.DataFrame({"hour": hours, "month": months, "driver": drivers})


def _hour_weights():
    w = np.ones(24) * 0.02
    for h in [19,20,21,22,23,0,1,2,3,4,5]:
        w[h] = 0.07
    w /= w.sum()
    return w

# ── Risk Algorithm ─────────────────────────────────────────────────────────────

def breach_probability(
    moon_phase: str,
    days_since_rain: int,
    elephant_activity: float,
    ndvi_anomaly: float,
    dist_water_km: float,
    historic_breaches: int,
    fence_condition: str,
) -> float:
    score = 0.0

    # Moon phase (elephants forage more on bright nights)
    moon_scores = {"New Moon": 10, "Waxing Crescent": 20, "First Quarter": 30,
                   "Waxing Gibbous": 45, "Full Moon": 60, "Waning Gibbous": 50,
                   "Last Quarter": 30, "Waning Crescent": 20}
    score += moon_scores.get(moon_phase, 25)

    # Rainfall drought stress
    score += min(days_since_rain * 0.8, 30)

    # Elephant activity index (1–10 scale → up to 25 pts)
    score += (elephant_activity / 10) * 25

    # NDVI anomaly: negative = scarce vegetation inside, pressure to move out
    if ndvi_anomaly < 0:
        score += abs(ndvi_anomaly) * 25

    # Proximity to water: closer water inside = lower pressure
    score -= max(0, (8 - dist_water_km) * 1.2)

    # Historic breach rate
    score += min(historic_breaches * 0.5, 15)

    # Fence condition
    condition_penalty = {"Good": 0, "Fair": 8, "Poor": 18}
    score += condition_penalty.get(fence_condition, 0)

    return round(min(max(score, 0), 100), 1)


def risk_label(prob: float):
    if prob >= 65:
        return "High", "#d62728"
    elif prob >= 35:
        return "Medium", "#ff7f0e"
    return "Low", "#2ca02c"

# ── Map ────────────────────────────────────────────────────────────────────────

def build_map(df: pd.DataFrame, selected_id: str | None = None) -> folium.Map:
    center = [0.35, 37.0]
    m = folium.Map(location=center, zoom_start=9,
                   tiles="CartoDB positron", control_scale=True)

    for _, row in df.iterrows():
        label, color = risk_label(row["base_prob"])
        radius = 10 if row["id"] == selected_id else 7
        popup_html = f"""
        <b>{row['name']}</b><br>
        Risk: <span style='color:{color}'><b>{label}</b></span><br>
        Base Prob: {row['base_prob']}%<br>
        Breaches (hist.): {row['historic_breaches']}<br>
        Fence: {row['fence_condition']}<br>
        NDVI Δ: {row['ndvi_anomaly']}<br>
        Water dist: {row['dist_water_km']} km
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['name']} ({label})",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-size:13px;">
      <b>Breach Risk</b><br>
      <span style='color:#d62728'>●</span> High (&ge;65%)<br>
      <span style='color:#ff7f0e'>●</span> Medium (35–64%)<br>
      <span style='color:#2ca02c'>●</span> Low (&lt;35%)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# ── Charts ─────────────────────────────────────────────────────────────────────

def hourly_breach_chart(history: pd.DataFrame):
    counts = history.groupby("hour").size().reset_index(name="breaches")
    counts["period"] = counts["hour"].apply(
        lambda h: "Night (18:00–06:00)" if h >= 18 or h < 6 else "Day (06:00–18:00)"
    )
    fig = px.bar(
        counts, x="hour", y="breaches", color="period",
        color_discrete_map={"Night (18:00–06:00)": "#3a0ca3", "Day (06:00–18:00)": "#f9c74f"},
        labels={"hour": "Hour of Day", "breaches": "Recorded Breaches"},
        title="Historical Breach Times (Hour of Day)",
    )
    fig.update_layout(legend_title_text="", margin=dict(t=40, b=20))
    return fig


def driver_donut(history: pd.DataFrame):
    counts = history["driver"].value_counts().reset_index()
    counts.columns = ["driver", "count"]
    fig = px.pie(counts, names="driver", values="count", hole=0.45,
                 title="Primary Environmental Drivers",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(margin=dict(t=40, b=20), legend=dict(font_size=11))
    return fig


def monthly_trend(history: pd.DataFrame):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    counts = history.groupby("month").size().reset_index(name="breaches")
    counts["month_name"] = counts["month"].apply(lambda x: month_names[x-1])
    fig = px.line(counts, x="month_name", y="breaches", markers=True,
                  title="Monthly Breach Frequency",
                  labels={"month_name": "Month", "breaches": "Breaches"})
    fig.update_traces(line_color="#e63946", marker_color="#e63946")
    fig.update_layout(margin=dict(t=40, b=20))
    return fig

# ── App Layout ─────────────────────────────────────────────────────────────────

def main():
    df       = generate_fence_data()
    history  = generate_breach_history()

    # Compute baseline probabilities with default environmental conditions
    df["base_prob"] = df.apply(lambda r: breach_probability(
        moon_phase="Waxing Gibbous",
        days_since_rain=18,
        elephant_activity=r["elephant_activity"],
        ndvi_anomaly=r["ndvi_anomaly"],
        dist_water_km=r["dist_water_km"],
        historic_breaches=r["historic_breaches"],
        fence_condition=r["fence_condition"],
    ), axis=1)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/320px-African_Bush_Elephant.jpg",
            use_container_width=True,
        )
        st.title("🐘 HEC Predictor")
        st.caption("Space for Giants — Laikipia Prototype")
        st.divider()

        st.subheader("Select Fence Segment")
        segment_names = df["name"].tolist()
        selected_name = st.selectbox("Fence Segment", segment_names, index=0)
        selected_row  = df[df["name"] == selected_name].iloc[0]

        st.divider()
        st.subheader("Environmental Conditions")

        moon_phase = st.select_slider(
            "Current Moon Phase",
            options=["New Moon","Waxing Crescent","First Quarter","Waxing Gibbous",
                     "Full Moon","Waning Gibbous","Last Quarter","Waning Crescent"],
            value="Waxing Gibbous",
        )
        days_since_rain = st.slider("Days Since Last Rainfall", 0, 60, 18)
        user_activity   = st.slider("Elephant Activity Level (observed)",
                                     1.0, 10.0, float(selected_row["elephant_activity"]), 0.5)

        st.divider()
        st.subheader("Segment Info")
        st.metric("Fence Condition", selected_row["fence_condition"])
        st.metric("Historic Breaches", selected_row["historic_breaches"])
        st.metric("NDVI Anomaly", selected_row["ndvi_anomaly"])
        st.metric("Dist. to Water Pan", f"{selected_row['dist_water_km']} km")

    # ── Compute live score ─────────────────────────────────────────────────────
    live_prob = breach_probability(
        moon_phase=moon_phase,
        days_since_rain=days_since_rain,
        elephant_activity=user_activity,
        ndvi_anomaly=float(selected_row["ndvi_anomaly"]),
        dist_water_km=float(selected_row["dist_water_km"]),
        historic_breaches=int(selected_row["historic_breaches"]),
        fence_condition=selected_row["fence_condition"],
    )
    risk, color = risk_label(live_prob)

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>Human-Elephant Conflict Prediction Dashboard</h1>"
        "<p style='color:gray;margin-top:2px'>Laikipia Ecosystem · Space for Giants</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── KPI Row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected Segment", selected_row["name"])
    k2.metric("Breach Probability", f"{live_prob}%",
              delta=f"{live_prob - selected_row['base_prob']:+.1f}% vs baseline")
    k3.metric("Risk Level", risk)
    k4.metric("Moon Phase", moon_phase)

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=live_prob,
        delta={"reference": selected_row["base_prob"], "valueformat": ".1f"},
        title={"text": f"Breach Probability — {selected_row['name']}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#d4edda"},
                {"range": [35, 65], "color": "#fff3cd"},
                {"range": [65,100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": live_prob},
        },
        number={"suffix": "%", "font": {"size": 36}},
    ))
    gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))

    col_gauge, col_map = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;padding:6px 12px;border-radius:6px;"
            f"background:{color};color:white;font-weight:bold;font-size:18px'>"
            f"{risk} RISK</div>",
            unsafe_allow_html=True,
        )

    # ── Map ────────────────────────────────────────────────────────────────────
    with col_map:
        m = build_map(df, selected_id=selected_row["id"])
        st_folium(m, width=None, height=400, returned_objects=[])

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.subheader("Historical Breach Analytics")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.plotly_chart(hourly_breach_chart(history), use_container_width=True)
    with ch2:
        st.plotly_chart(driver_donut(history), use_container_width=True)
    with ch3:
        st.plotly_chart(monthly_trend(history), use_container_width=True)

    st.divider()

    # ── Risk Table ─────────────────────────────────────────────────────────────
    st.subheader("All Fence Segments — Risk Overview")
    display = df[["name","historic_breaches","dist_water_km","ndvi_anomaly",
                  "elephant_activity","fence_condition","base_prob"]].copy()
    display.columns = ["Segment","Hist. Breaches","Water (km)",
                       "NDVI Δ","Elephant Activity","Fence","Base Prob %"]
    display = display.sort_values("Base Prob %", ascending=False).reset_index(drop=True)

    def color_risk(val):
        _, c = risk_label(val)
        return f"background-color:{c};color:white;font-weight:bold"

    st.dataframe(
        display.style.map(color_risk, subset=["Base Prob %"]),
        use_container_width=True,
        height=420,
    )

    st.caption(
        "Prototype built for Space for Giants · Laikipia, Kenya · "
        f"Data refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


if __name__ == "__main__":
    main()
