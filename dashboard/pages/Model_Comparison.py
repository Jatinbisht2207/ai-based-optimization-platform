import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json


st.markdown(
    "<h1 style='text-align:center;'> Model Performance Intelligence</h1>",
    unsafe_allow_html=True
)

# ===============================
# LOAD METRICS
# ===============================
with open("outputs/model_metrics.json", "r") as f:
    metrics = json.load(f)

metrics_df = pd.DataFrame(metrics).T
metrics_df = metrics_df.round(4)


# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("Comparison Controls")

view_option = st.sidebar.radio(
    "Select View",
    [
        "Metric Table",
        "MAE Comparison",
        "RMSE Comparison",
        "R² Comparison",
        "Best Model Summary"
    ]
)

# ===============================
# KPI SUMMARY
# ===============================
best_mae_model = metrics_df["MAE"].idxmin()
best_rmse_model = metrics_df["RMSE"].idxmin()
best_r2_model = metrics_df["R2"].idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("Best MAE Model", best_mae_model)
col2.metric("Best RMSE Model", best_rmse_model)
col3.metric("Best R² Model", best_r2_model)

st.markdown("---")


# ===============================
# VIEW SWITCH
# ===============================
if view_option == "Metric Table":

    st.dataframe(metrics_df, use_container_width=True)


elif view_option == "MAE Comparison":

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics_df.index, metrics_df["MAE"])
    ax.set_title("MAE Comparison")
    ax.set_ylabel("MAE")
    st.pyplot(fig)


elif view_option == "RMSE Comparison":

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics_df.index, metrics_df["RMSE"])
    ax.set_title("RMSE Comparison")
    ax.set_ylabel("RMSE")
    st.pyplot(fig)


elif view_option == "R² Comparison":

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics_df.index, metrics_df["R2"])
    ax.set_title("R² Comparison")
    ax.set_ylabel("R²")
    st.pyplot(fig)


elif view_option == "Best Model Summary":

    # Weighted scoring system
    metrics_df["Score"] = (
        (metrics_df["R2"] * 0.4) -
        (metrics_df["MAE"] * 0.3) -
        (metrics_df["RMSE"] * 0.3)
    )

    best_model = metrics_df["Score"].idxmax()

    st.markdown(f"##  Overall Best Model: **{best_model}**")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics_df.index, metrics_df["Score"])
    ax.set_title("Composite Model Score")
    st.pyplot(fig)