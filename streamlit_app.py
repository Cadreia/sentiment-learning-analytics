import streamlit as st
import nltk
import spacy
import subprocess

nltk.download("punkt")
nltk.download("stopwords")


# Check if model is installed; if not, download it
def ensure_spacy_model(model_name):
    if not spacy.util.is_package(model_name):
        subprocess.run(["python", "-m", "spacy", "download", model_name])


ensure_spacy_model("en_core_web_md")

# Streamlit page configuration for faster loading
st.set_page_config(page_title="Sentiment Analysis & Learning Analytics", layout="centered")


# Check if already integrated data exists
def load_integrated_data():
    if st.session_state.get('has_analytics', False) and st.session_state.get('has_feedback', False):
        return True
    return None


# App UI

# Define the navigation structure with pages grouped into categories
pages = {
    "Student Insights": [
        st.Page("views/overview.py", title="Overview"),
        st.Page("views/data_upload.py", title="Data Upload"),
        st.Page("views/view_all_data.py", title="View All Data"),
        st.Page("views/view_all_predictions.py", title="View All Predictions"),
        st.Page("views/analyze_student.py", title="Analyze Student"),
    ],
    "Analysis Tools": [
        st.Page("views/shap_explanations.py", title="SHAP Explanations"),
        st.Page("views/cluster_visualization.py", title="Cluster Visualization"),
        st.Page("views/agent_actions.py", title="Agent Actions"),
    ],
}

# Set up navigation and run the selected page
pg = st.navigation(pages)
pg.run()
