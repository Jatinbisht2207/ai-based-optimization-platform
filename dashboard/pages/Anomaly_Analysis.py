import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.markdown(
    "<h1 style='text-align:center;'>Anomaly Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)


# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("Anomaly Controls")

model = st.sidebar.selectbox(
    "Select Model",
    ["Regression", "Random Forest", "ARIMA", "SARIMA"]
)

threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=1.5,
    max_value=4.0,
    value=2.5,
    step=0.1
)

view_option = st.sidebar.radio(
    "Select View",
    [
        "Time Series",
        "Rolling Density",
        "Residual Distribution",
        "Severity Breakdown"
    ]
)


# ===============================
# LOAD DATA
# ===============================
file_map = {
    "Regression": "outputs/regression_predictions.csv",
    "Random Forest": "outputs/random_forest_predictions.csv",
    "ARIMA": "outputs/arima_predictions.csv",
    "SARIMA": "outputs/sarima_predictions.csv"
}

df = pd.read_csv(file_map[model])
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")


# ===============================
# ANOMALY COMPUTATION
# ===============================
df["Residual"] = df["Actual"] - df["Predicted"]
df["Z_score"] = (df["Residual"] - df["Residual"].mean()) / df["Residual"].std()
df["Anomaly"] = np.abs(df["Z_score"]) > threshold

total_anomalies = df["Anomaly"].sum()
anomaly_percent = (total_anomalies / len(df)) * 100


# ===============================
# SYSTEM STATUS
# ===============================
if anomaly_percent < 3:
    status = "System Stable"
elif anomaly_percent < 7:
    status = "Moderate Instability"
else:
    status = "High Anomaly Risk"

col1, col2, col3 = st.columns(3)
col1.metric("Total Anomalies", int(total_anomalies))
col2.metric("Anomaly %", f"{anomaly_percent:.2f}%")
col3.metric("System Status", status)

st.markdown("---")


# ===============================
# VISUALIZATION SWITCH
# ===============================

if view_option == "Time Series":

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(df["Timestamp"], df["Actual"], linewidth=1)

    ax.scatter(
        df["Timestamp"][df["Anomaly"]],
        df["Actual"][df["Anomaly"]],
        color="red",
        s=25
    )

    ax.set_title(f"{model} - Time-Based Anomaly Detection")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)


elif view_option == "Rolling Density":

    df["Anomaly_Int"] = df["Anomaly"].astype(int)
    df["Rolling_Anomaly_Rate"] = df["Anomaly_Int"].rolling(48).mean()

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(df["Timestamp"], df["Rolling_Anomaly_Rate"])

    ax.set_title("Rolling Anomaly Density (Daily Window)")
    ax.set_ylabel("Anomaly Rate")
    ax.set_xlabel("Time")

    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)


elif view_option == "Residual Distribution":

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(df["Residual"], bins=40)

    mean = df["Residual"].mean()
    std = df["Residual"].std()

    ax.axvline(mean, linestyle="--")
    ax.axvline(mean + threshold * std, linestyle="--")
    ax.axvline(mean - threshold * std, linestyle="--")

    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)


elif view_option == "Severity Breakdown":

    conditions = [
        (np.abs(df["Z_score"]) > threshold) & (np.abs(df["Z_score"]) <= threshold + 0.5),
        (np.abs(df["Z_score"]) > threshold + 0.5) & (np.abs(df["Z_score"]) <= threshold + 1),
        (np.abs(df["Z_score"]) > threshold + 1)
    ]

    labels = ["Mild", "Moderate", "Severe"]

    df["Severity"] = np.select(conditions, labels, default="Normal")

    severity_counts = df["Severity"].value_counts().reindex(
        ["Normal", "Mild", "Moderate", "Severe"]
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(7,4))

    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    ax.bar(
        severity_counts.index,
        severity_counts.values,
        color=colors
    )

    ax.set_title("Anomaly Severity Breakdown")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Number of Observations")

    for i, v in enumerate(severity_counts.values):
        ax.text(i, v + 1, str(int(v)), ha='center')

    st.pyplot(fig)


# ===============================
# ANOMALY TABLE
# ===============================
if st.sidebar.checkbox("Show Anomaly Table"):

    anomaly_table = df[df["Anomaly"]][
        ["Timestamp", "Actual", "Predicted", "Residual", "Z_score"]
    ].copy()

    anomaly_table["Timestamp"] = anomaly_table["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    st.markdown("### Detected Anomalies")
    st.dataframe(anomaly_table, use_container_width=True)