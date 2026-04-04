def render_page():
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

    df["Timestamp"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="30min"
    )

    df["Residual"] = df["Actual"] - df["Predicted"]
    df["Z_score"] = (df["Residual"] - df["Residual"].mean()) / df["Residual"].std()
    df["Anomaly"] = np.abs(df["Z_score"]) > z_threshold
    df["Energy_Wastage"] = np.where(df["Anomaly"], np.abs(df["Residual"]), 0)

    total_anomalies = int(df["Anomaly"].sum())
    total_wastage = float(df["Energy_Wastage"].sum())
    total_actual = float(df["Actual"].sum())

    efficiency = max(0, 1 - (total_wastage / total_actual))
    eff_percent = round(efficiency * 100, 1)

    estimated_cost = total_wastage * tariff

    # ==========================================
    # PAGE TITLE
    # ==========================================
    st.markdown("##  System Insights")

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
            number={"suffix": "%", "font": {"size": 48, "color": "#2b6cb0"}},
            title={"text": "System Efficiency"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3182ce"},
                "steps": [
                    {"range": [0, 70], "color": "#fde2e2"},
                    {"range": [70, 85], "color": "#fff4d6"},
                    {"range": [85, 100], "color": "#e6f4ea"}
                ],
            }
        ))

        fig.update_layout(height=380)
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

        # ==========================================
        # 🧠 DECISION LAYER
        # ==========================================
        st.markdown("###  AI Decision Insights")

        anomaly_times = df[df["Anomaly"]]["Timestamp"]
        peak_hour = (
            anomaly_times.dt.hour.value_counts().idxmax()
            if len(anomaly_times) > 0 else "N/A"
        )

        df["Date"] = df["Timestamp"].dt.date
        daily_wastage = df.groupby("Date")["Energy_Wastage"].sum()
        worst_day = (
            str(daily_wastage.idxmax())
            if len(daily_wastage) > 0 else "N/A"
        )

        model_scores = {}
        for model in ["Regression", "Random Forest", "ARIMA", "SARIMA"]:
            temp_df = load_predictions(model)
            mae = np.mean(np.abs(temp_df["Actual"] - temp_df["Predicted"]))
            model_scores[model] = mae

        best_model = min(model_scores, key=model_scores.get)

        st.write(f" Peak Anomaly Hour: {peak_hour}")
        st.write(f" Most Inefficient Day: {worst_day}")
        st.write(f" Best Model: {best_model}")

        st.markdown("---")

        st.markdown("###  Recommendations")

        if isinstance(peak_hour, int) and 17 <= peak_hour <= 22:
            st.success("Reduce energy usage during evening peak hours")

        if total_wastage > 0:
            st.success("Optimize load distribution to reduce wastage")

        st.success(f"Use {best_model} model for best accuracy")

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