import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_predictions(model_name):
    file_map = {
        "Regression": "outputs/regression_predictions.csv",
        "Random Forest": "outputs/random_forest_predictions.csv",
        "ARIMA": "outputs/arima_predictions.csv",
        "SARIMA": "outputs/sarima_predictions.csv"
    }
    return pd.read_csv(file_map[model_name])


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙ System Configuration")

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Regression", "Random Forest", "ARIMA", "SARIMA"]
)

tariff = st.sidebar.number_input(
    "Electricity Tariff (₹ per unit)",
    min_value=1.0,
    max_value=20.0,
    value=8.0,
    step=0.5
)

z_threshold = st.sidebar.slider(
    "Anomaly Z-Score Threshold",
    1.5, 4.0, 2.5, 0.1
)

# ==========================================
# PROCESS DATA
# ==========================================
df = load_predictions(selected_model)

df["Residual"] = df["Actual"] - df["Predicted"]
df["Z_score"] = (df["Residual"] - df["Residual"].mean()) / df["Residual"].std()
df["Anomaly"] = np.abs(df["Z_score"]) > z_threshold
df["Energy_Wastage"] = np.where(df["Anomaly"], np.abs(df["Residual"]), 0)

total_anomalies = int(df["Anomaly"].sum())
total_wastage = float(df["Energy_Wastage"].sum())
total_actual = float(df["Actual"].sum())

# TRUE Efficiency
efficiency = max(0, 1 - (total_wastage / total_actual))
eff_percent = round(efficiency * 100, 1)

estimated_cost = total_wastage * tariff

# ==========================================
# PAGE TITLE
# ==========================================
st.markdown("##  System Insights")

# KPI Row
c1, c2, c3 = st.columns(3)
c1.metric("Total Anomalies", total_anomalies)
c2.metric("Total Energy Wastage", round(total_wastage, 2))
c3.metric("System Efficiency", round(efficiency, 3))

st.divider()

# ==========================================
# LAYOUT
# ==========================================
left_space, col_gauge, col_info = st.columns([0.2, 2, 1])

# ---- GAUGE ----
with col_gauge:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eff_percent,
        number={
            "suffix": "%",
            "font": {"size": 48, "color": "#2b6cb0"}
        },
        title={
            "text": "System Efficiency",
            "font": {"size": 20}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#999"
            },
            "bar": {
                "color": "#3182ce",
                "thickness": 0.20
            },
            "steps": [
                {"range": [0, 70], "color": "#fde2e2"},
                {"range": [70, 85], "color": "#fff4d6"},
                {"range": [85, 100], "color": "#e6f4ea"}
            ],
            "threshold": {
                "line": {"color": "#1f4e79", "width": 4},
                "thickness": 0.9,
                "value": eff_percent
            },
            "bgcolor": "white",
            "borderwidth": 0
        }
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=10),
        transition={'duration': 500}
    )

    st.plotly_chart(fig, use_container_width=True)


# ---- RIGHT PANEL ----
with col_info:

    st.markdown("### Risk Classification")

    if efficiency >= 0.85:
        risk_status = "🟢 Stable"
    elif efficiency >= 0.70:
        risk_status = "🟡 Moderate"
    else:
        risk_status = "🔴 Critical"

    st.markdown(f"### {risk_status}")

    st.markdown("---")

    st.markdown("###  Financial Impact")
    st.write(f"Tariff Applied: ₹{tariff}")
    st.write(f"Estimated Energy Loss Cost: ₹{round(estimated_cost, 2)}")

    st.markdown("---")

    st.download_button(
        " Export Financial Report",
        pd.DataFrame({
            "Model Used": [selected_model],
            "Total Anomalies": [total_anomalies],
            "Total Wastage": [round(total_wastage, 2)],
            "System Efficiency": [efficiency],
            "Tariff": [tariff],
            "Estimated Cost": [round(estimated_cost, 2)]
        }).to_csv(index=False),
        "financial_report.csv",
        "text/csv"
    )
 