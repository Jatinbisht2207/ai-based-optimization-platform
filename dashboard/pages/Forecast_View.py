def render_page():
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    from forecasting.future_predictor import predict_future_consumption
    from forecasting.future_predictor_v2 import predict_full_day
    from forecasting.lstm_model import predict_lstm

    from sklearn.metrics import mean_absolute_error, mean_squared_error


    # =============================
    # PAGE CONFIG
    # =============================
    st.set_page_config(layout="wide")


    # =============================
    # SESSION STATE
    # =============================
    if "page_mode" not in st.session_state:
        st.session_state.page_mode = "forecast"


    # =============================
    # SIDEBAR
    # =============================
    st.sidebar.markdown("## ⚙ Forecast Controls")

    aggregation = st.sidebar.selectbox(
        "Aggregation Level",
        ["Detailed (30 Min)", "Hourly", "Daily"]
    )

    # 🔥 HIDE MODEL SELECTOR IN PREDICTION MODE
    if st.session_state.page_mode != "prediction":
        model_selector = st.sidebar.selectbox(
            "Forecast Model",
            ["Regression", "Random Forest", "ARIMA", "SARIMA", "LSTM"]
        )
    else:
        model_selector = "Random Forest"

    visual_type = st.sidebar.selectbox(
        "Visualization",
        ["Line Chart", "Bar Chart", "Pie Chart"]
    )

    if st.session_state.page_mode == "forecast":
        if st.sidebar.button("Prediction"):
            st.session_state.page_mode = "prediction"
            st.rerun()
    else:
        if st.sidebar.button("Back to Forecast"):
            st.session_state.page_mode = "forecast"
            st.rerun()


    # =============================
    # 🔮 PREDICTION PAGE
    # =============================
    if st.session_state.page_mode == "prediction":

        st.title("Future Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Timestamp Prediction")

            selected_date = st.date_input("Select Date")
            selected_time = st.time_input("Select Time")

            if st.button("Run Timestamp Prediction"):
                timestamp = pd.Timestamp.combine(selected_date, selected_time)

                try:
                    pred = predict_future_consumption(timestamp)
                    st.success("Prediction Generated Successfully")
                    st.metric("Predicted Consumption", f"{pred:.4f}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        with col2:
            st.subheader("Day Prediction")

            selected_day = st.date_input("Select Day", key="day_pred")

            if st.button("Run Day Prediction"):
                try:
                    future_df = predict_full_day(selected_day)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=future_df["Timestamp"],
                        y=future_df["Predicted_Consumption"],
                        mode="lines",
                        name="Predicted",
                        line=dict(color="red")
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        st.stop()


    # =============================
    # LSTM MODE
    # =============================
    if model_selector == "LSTM":

        st.title("LSTM Forecast Intelligence")

        try:
            df = predict_lstm()

            st.success("LSTM Prediction Generated Successfully")

            mae = mean_absolute_error(df["Electricity_Consumed"], df["Predicted"])
            rmse = np.sqrt(mean_squared_error(df["Electricity_Consumed"], df["Predicted"]))
            accuracy = 100 - (mae / df["Electricity_Consumed"].mean() * 100)

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric("Accuracy", f"{accuracy:.2f}%")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=df["Electricity_Consumed"],
                mode="lines",
                name="Actual",
                line=dict(color="blue")
            ))

            fig.add_trace(go.Scatter(
                y=df["Predicted"],
                mode="lines",
                name="LSTM Predicted",
                line=dict(color="red")
            ))

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"LSTM Error: {e}")

        st.stop()


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


    if aggregation == "Hourly":
        df_plot = df.resample("H").mean()
    elif aggregation == "Daily":
        df_plot = df.resample("D").mean()
    else:
        df_plot = df.copy()


    mae = mean_absolute_error(df_plot["Actual"], df_plot["Predicted"])
    rmse = np.sqrt(mean_squared_error(df_plot["Actual"], df_plot["Predicted"]))
    accuracy = 100 - (mae / df_plot["Actual"].mean() * 100)


    st.markdown(
        "<h1 style='text-align:center;'>Advanced Forecast Intelligence</h1>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("Accuracy", f"{accuracy:.2f}%")


    if accuracy > 85:
        st.success("🟢 Excellent Forecast Performance")
    elif accuracy > 70:
        st.info("🟡 Moderate Forecast Performance")
    else:
        st.warning("🔴 Model Requires Optimization")


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

        st.plotly_chart(fig, use_container_width=True)


    elif visual_type == "Bar Chart":

        fig = go.Figure(data=[
            go.Bar(name="Actual Avg", x=["Actual"], y=[df_plot["Actual"].mean()], marker_color="blue"),
            go.Bar(name="Predicted Avg", x=["Predicted"], y=[df_plot["Predicted"].mean()], marker_color="red")
        ])

        st.plotly_chart(fig, use_container_width=True)


    elif visual_type == "Pie Chart":

        fig = go.Figure(data=[go.Pie(
            labels=["Actual Total", "Predicted Total"],
            values=[df_plot["Actual"].sum(), df_plot["Predicted"].sum()],
            hole=0.4
        )])

        st.plotly_chart(fig, use_container_width=True)