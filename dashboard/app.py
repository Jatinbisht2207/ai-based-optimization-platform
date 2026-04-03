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
from pages.Forecast_View import render_page as forecast
from pages.Model_Comparison import render_page as model_comparison
from pages.System_Insights import render_page as system_insights
from pages.Wastage_Analysis import render_page as wastage
from pages.Anomaly_Analysis import render_page as anomaly

# ===============================
# SIDEBAR
# ===============================
st.title(" AI-Based Smart Energy Optimization Platform")
st.markdown("---")
st.sidebar.title(" Energy Optimization App")

pages = {
    " Forecast View": forecast,
    " Model Comparison": model_comparison,
    " System Insights": system_insights,
    " Wastage Analysis": wastage,
    " Anomaly Analysis": anomaly
}

selected_page = st.sidebar.radio("Navigate", list(pages.keys()))

# ===============================
# RENDER SELECTED PAGE
# ===============================
pages[selected_page]()