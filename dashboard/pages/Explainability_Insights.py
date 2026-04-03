def render_page():
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import os

    st.set_page_config(page_title="Explainability Insights", layout="wide")

    st.markdown("<h1 style='margin-top:-30px;'> Explainability Insights</h1>", unsafe_allow_html=True)

    st.sidebar.header("Explainability Controls")

    top_n = st.sidebar.slider(
        "Top N Features",
        min_value=5,
        max_value=20,
        value=10
    )

    # --------------------------------------------------
    # Absolute Path Fix
    # --------------------------------------------------

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    file_path = os.path.join(BASE_DIR, "outputs", "rf_feature_importance.csv")

    if not os.path.exists(file_path):
        st.warning("Feature importance file not found. Run main.py first.")
        st.stop()

    importance_df = pd.read_csv(file_path)

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    st.subheader(" Global Feature Importance (Random Forest)")

    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

