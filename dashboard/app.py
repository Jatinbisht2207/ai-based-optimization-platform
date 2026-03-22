import streamlit as st

# Import all page render functions
from pages.Forecast_View import render_page as render_forecast
from pages.Model_Comparison import render_page as render_model_comparison
from pages.System_Insights import render_page as render_system_insights
from pages.Wastage_Analysis import render_page as render_wastage_analysis
from pages.Anomaly_Analysis import render_page as render_anomaly_analysis


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Smart Energy Optimization",
    layout="wide"
)

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("app")

page = st.sidebar.radio(
    "Navigate",
    [
        "Anomaly Analysis",
        "Forecast View",
        "Model Comparison",
        "System Insights",
        "Wastage Analysis"
    ]
)

# ===============================
# ROUTING
# ===============================
if page == "Forecast View":
    render_forecast()

elif page == "Model Comparison":
    render_model_comparison()

elif page == "System Insights":
    render_system_insights()

elif page == "Wastage Analysis":
    render_wastage_analysis()

elif page == "Anomaly Analysis":
    render_anomaly_analysis()