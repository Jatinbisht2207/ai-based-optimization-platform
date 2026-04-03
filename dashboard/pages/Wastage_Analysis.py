def render_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px

    st.set_page_config(page_title="Wastage Analysis", layout="wide")

    # -------- REMOVE EXTRA TOP SPACE ----------
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        margin-bottom: 0.3rem;
    }
    hr {
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1> Energy Wastage Analysis</h1>", unsafe_allow_html=True)

    # ======================================================
    # SIDEBAR CONTROLS
    # ======================================================

    st.sidebar.header("Wastage Controls")

    model_option = st.sidebar.selectbox(
        "Select Forecast Model",
        ("Regression", "Random Forest", "ARIMA", "SARIMA")
    )

    critical_threshold = st.sidebar.slider(
        "Critical Wastage Threshold",
        0.0, 1.0, 0.25, 0.05
    )

    tariff = st.sidebar.number_input(
        "Energy Tariff (₹ per unit)",
        min_value=0.0,
        value=8.0
    )

    top_n = st.sidebar.slider(
        "Top N Worst Events",
        3, 15, 5
    )

    section = st.sidebar.radio(
        "View Section",
        (
            "Trend",
            "Cumulative",
            "Hourly",
            "Risk",
            "Top Events",
            "Financial"
        )
    )

    file_map = {
        "Regression": "outputs/regression_predictions.csv",
        "Random Forest": "outputs/random_forest_predictions.csv",
        "ARIMA": "outputs/arima_predictions.csv",
        "SARIMA": "outputs/sarima_predictions.csv",
    }

    # ======================================================
    # LOAD DATA
    # ======================================================

    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    df = load_data(file_map[model_option])

    # ======================================================
    # COLUMN DETECTION
    # ======================================================

    possible_actual_cols = ["Electricity_Consumed", "Actual", "y_test"]
    possible_pred_cols = ["Predicted", "Prediction", "y_pred"]

    actual_col = next((c for c in possible_actual_cols if c in df.columns), None)
    pred_col = next((c for c in possible_pred_cols if c in df.columns), None)

    if actual_col is None or pred_col is None:
        st.error("Required columns not found.")
        st.write(df.columns)
        st.stop()

    # ======================================================
    # CALCULATIONS
    # ======================================================

    df["Wastage"] = np.where(
        df[actual_col] > df[pred_col],
        df[actual_col] - df[pred_col],
        0
    )

    df["Hour"] = df["Timestamp"].dt.hour

    total_consumption = df[actual_col].sum()
    total_wastage = df["Wastage"].sum()
    avg_wastage = df["Wastage"].mean()
    wastage_percent = (total_wastage / total_consumption) * 100
    critical_percent = (
        len(df[df["Wastage"] > critical_threshold]) / len(df)
    ) * 100

    # ======================================================
    # KPI SECTION
    # ======================================================

    st.markdown(f"###  Model: {model_option}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Wastage", f"{total_wastage:.2f} units")
    col2.metric("Wastage %", f"{wastage_percent:.2f}%")
    col3.metric("Avg / Interval", f"{avg_wastage:.4f}")
    col4.metric("Critical %", f"{critical_percent:.2f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ======================================================
    # SECTION RENDERING
    # ======================================================

    if section == "Trend":

        fig = px.line(df, x="Timestamp", y="Wastage")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Cumulative":

        df["Cumulative_Wastage"] = df["Wastage"].cumsum()
        fig = px.line(df, x="Timestamp", y="Cumulative_Wastage")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Hourly":

        hourly = df.groupby("Hour")["Wastage"].sum().reset_index()
        fig = px.bar(hourly, x="Hour", y="Wastage")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Risk":

        def classify_risk(val):
            if val < 0.1:
                return "Low"
            elif val < 0.3:
                return "Moderate"
            else:
                return "High"

        df["Risk_Level"] = df["Wastage"].apply(classify_risk)
        risk_counts = df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]

        fig = px.pie(risk_counts, names="Risk_Level", values="Count")
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Top Events":

        top_events = df.sort_values("Wastage", ascending=False).head(top_n)
        st.dataframe(
            top_events[["Timestamp", actual_col, pred_col, "Wastage"]],
            use_container_width=True
        )

    elif section == "Financial":

        current_loss = total_wastage * tariff
        save_20 = current_loss * 0.20
        save_30 = current_loss * 0.30

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Loss", f"₹ {current_loss:.2f}")
        col2.metric("Savings (20%)", f"₹ {save_20:.2f}")
        col3.metric("Savings (30%)", f"₹ {save_30:.2f}")