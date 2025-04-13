# pages/shap_explanations.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

def page():
    st.title("SHAP Explanations")

    # Check if model and data are available
    if 'model' not in st.session_state or 'reduced_features' not in st.session_state:
        st.warning("No model or data available. Please run the analysis on the Overview page first.")
        return

    # Retrieve model and data
    model = st.session_state['model']
    X = st.session_state['reduced_features']
    data = st.session_state['integrated_data']

    # Compute SHAP values
    st.write("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X, feature_names=[f"Feature {i}" for i in range(X.shape[1])], show=False)
    st.pyplot(fig)

    # Individual student explanation
    st.subheader("Individual Student Explanation")
    student_ids = data['Student_ID'].astype(str).tolist()
    selected_student = st.selectbox("Select a Student ID:", student_ids)
    student_idx = data.index[data['Student_ID'].astype(str) == selected_student][0]

    st.write(f"SHAP Force Plot for Student ID: {selected_student}")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][student_idx], X[student_idx], feature_names=[f"Feature {i}" for i in range(X.shape[1])], matplotlib=True)
    st.pyplot(plt.gcf())

page()