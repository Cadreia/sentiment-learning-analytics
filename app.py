# app.py
import streamlit as st

# Streamlit page configuration for faster loading
st.set_page_config(page_title="Sentiment Analysis & Learning Analytics", layout="centered")

# Check if already integrated data exists
def load_integrated_data():
    if st.session_state.get('has_analytics', False) and st.session_state.get('has_feedback', False):
        return True
    return None


# if load_integrated_data() is None:
#     st.warning(
#         "Please upload both Analytics Data and Feedback Data using the Data Upload page before proceeding.")

# App UI

# Define the navigation structure with pages grouped into categories
pages = {
    "Student Insights": [
        st.Page("pages/overview.py", title="Overview"),
        st.Page("pages/data_upload.py", title="Data Upload"),
        st.Page("pages/view_all_data.py", title="View All Data"),
        st.Page("pages/view_all_predictions.py", title="View All Predictions"),
        st.Page("pages/analyze_student.py", title="Analyze Student"),
    ],
    "Analysis Tools": [
        st.Page("pages/shap_explanations.py", title="SHAP Explanations"),
        st.Page("pages/cluster_visualization.py", title="Cluster Visualization"),
        st.Page("pages/agent_actions.py", title="Agent Actions"),
    ],
}

# Set up navigation and run the selected page
pg = st.navigation(pages)
pg.run()
