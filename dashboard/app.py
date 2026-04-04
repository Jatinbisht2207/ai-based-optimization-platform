import streamlit as st

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Smart Energy Optimization",
    layout="wide"
)

# ===============================
# IMPORT PAGES
# ===============================
from my_pages.Forecast_View import render_page as forecast
from my_pages.Explainability_Insights import render_page as explainability
from my_pages.Model_Comparison import render_page as model_comparison
from my_pages.System_Insights import render_page as system_insights
from my_pages.Wastage_Analysis import render_page as wastage
from my_pages.Anomaly_Analysis import render_page as anomaly

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title(" Energy Optimization App")
st.sidebar.markdown("---")

pages = {
    " Home": "home",
    " Anomaly Analysis": anomaly,
    " Explainability Insights": explainability,
    " Forecast View": forecast,
    " Model Comparison": model_comparison,
    " System Insights": system_insights,
    " Wastage Analysis": wastage
}

selected_page = st.sidebar.radio(" Navigate", list(pages.keys()))

st.sidebar.markdown("---")

# ===============================
# HOME PAGE
# ===============================
if selected_page == " Home":

    st.title(" AI-Based Smart Energy Optimization Platform")
    st.markdown("---")

    st.markdown("""
    ###  Project Overview

    This platform is designed to intelligently analyze and optimize energy consumption using AI.

    ###  Key Features
    -  Energy Consumption Forecasting  
    -  Machine Learning Models (RF, Regression, ARIMA)  
    -  Anomaly Detection System  
    -  Energy Wastage Analysis  
    -  Explainable AI Insights  
    -  Future Energy Prediction  

    ### 🧱Architecture
    - Modular pipeline (Ingestion → Preprocessing → Forecasting → Visualization)
    - Built using Python + Streamlit
    - Designed for real-time analytics

    ###  Goal
    To provide an intelligent system for monitoring, predicting, and optimizing energy usage efficiently.
    """)

# ===============================
# RENDER OTHER PAGES
# ===============================
else:
    pages[selected_page]()