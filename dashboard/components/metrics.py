import streamlit as st

def display_kpis(total_anomalies, total_wastage, efficiency):
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Anomalies", total_anomalies)
    col2.metric("Total Energy Wastage", f"{total_wastage:.2f}")
    col3.metric("System Efficiency", f"{efficiency:.3f}")