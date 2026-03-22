import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from forecasting.future_predictor import predict_future_consumption
from forecasting.future_predictor_v2 import predict_full_day
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================
# SESSION STATE
# =============================
if "future_mode" not in st.session_state:
    st.session_state.future_mode = False

if "prediction_type" not in st.session_state:
    st.session_state.prediction_type = None


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(layout="wide")


# =============================
# SIDEBAR
# =============================
st.sidebar.markdown("## ⚙ Forecast Controls")

aggregation = st.sidebar.selectbox(
    "Aggregation Level",
    ["Detailed (30 Min)", "Hourly", "Daily"]
)

model_selector = st.sidebar.selectbox(
    "Forecast Model",
    ["Regression", "Random Forest", "ARIMA", "SARIMA"]
)

visual_type = st.sidebar.selectbox(
    "Visualization",
    ["Line Chart", "Bar Chart", "Pie Chart"]
)

# FUTURE FORECAST BUTTON
if st.sidebar.button(" Future Energy Forecast"):
    st.session_state.future_mode = True
    st.session_state.prediction_type = None


# =====================================================
# FUTURE FORECAST MODE (OVERRIDES WHOLE PAGE)
# =====================================================
if st.session_state.future_mode:

    # Top header with back button
    col_title, col_back = st.columns([8,2])

    with col_title:
        st.title("🔮 Future Energy Forecast")

    with col_back:
        if st.button("⬅ Back to Forecast Intelligence"):
            st.session_state.future_mode = False
            st.session_state.prediction_type = None
            st.rerun()

    col1, col2 = st.columns(2)

    if col1.button("Timestamp Prediction"):
        st.session_state.prediction_type = "timestamp"

    if col2.button("Full Day Prediction"):
        st.session_state.prediction_type = "fullday"


# =============================
# TIMESTAMP PREDICTION
# =============================
    if st.session_state.prediction_type == "timestamp":

        st.subheader("Timestamp Energy Prediction")

        future_datetime = st.datetime_input(
            "Select Future Date & Time"
        )

        if st.button("Predict Energy"):

            try:

                prediction = predict_future_consumption(future_datetime)

                st.success(
                    f"Predicted Energy Consumption: {prediction:.4f} kWh"
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# =============================
# FULL DAY FORECAST
# =============================
    if st.session_state.prediction_type == "fullday":

        st.subheader("24 Hour Energy Forecast")

        future_date = st.date_input("Select Future Date")

        if st.button("Generate Forecast"):

            try:

                df_future = predict_full_day(future_date)

                st.dataframe(df_future)

                fig_future = go.Figure()

                fig_future.add_trace(go.Scatter(
                    x=df_future["Timestamp"],
                    y=df_future["Predicted_Consumption"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red")
                ))

                fig_future.update_layout(
                    title="Future Energy Demand Curve",
                    height=450
                )

                st.plotly_chart(fig_future, use_container_width=True)

                peak_row = df_future.loc[
                    df_future["Predicted_Consumption"].idxmax()
                ]

                low_row = df_future.loc[
                    df_future["Predicted_Consumption"].idxmin()
                ]

                st.info(
                    f" Peak Demand: {peak_row['Timestamp']}  \n"
                    f" Lowest Demand: {low_row['Timestamp']}"
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# =====================================================
# NORMAL FORECAST DASHBOARD
# =====================================================
else:

    # =============================
    # LOAD DATA
    # =============================
    model_file_map = {
        "Regression": "outputs/regression_predictions.csv",
        "Random Forest": "outputs/random_forest_predictions.csv",
        "ARIMA": "outputs/arima_predictions.csv",
        "SARIMA": "outputs/sarima_predictions.csv"
    }

    df = pd.read_csv(model_file_map[model_selector])

    df["Timestamp"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="30T"
    )

    df.set_index("Timestamp", inplace=True)


    # =============================
    # AGGREGATION
    # =============================
    if aggregation == "Hourly":
        df_plot = df.resample("H").mean()
    elif aggregation == "Daily":
        df_plot = df.resample("D").mean()
    else:
        df_plot = df.copy()


    # =============================
    # METRICS
    # =============================
    mae = mean_absolute_error(df_plot["Actual"], df_plot["Predicted"])
    rmse = np.sqrt(mean_squared_error(df_plot["Actual"], df_plot["Predicted"]))
    accuracy = 100 - (mae / df_plot["Actual"].mean() * 100)


    # =============================
    # TITLE
    # =============================
    st.markdown(
        "<h1 style='text-align:center;'>Advanced Forecast Intelligence</h1>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("Accuracy", f"{accuracy:.2f}%")


    # =============================
    # STATUS
    # =============================
    if accuracy > 85:
        st.success("🟢 Excellent Forecast Performance")
    elif accuracy > 70:
        st.info("🟡 Moderate Forecast Performance")
    else:
        st.warning("🔴 Model Requires Optimization")


    # =============================
    # FORECAST GRAPH
    # =============================
    if visual_type == "Line Chart":

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="blue")
        ))

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot["Predicted"],
            mode="lines",
            name="Predicted",
            line=dict(color="red")
        ))

        fig.update_layout(height=420)

        st.plotly_chart(fig, use_container_width=True)