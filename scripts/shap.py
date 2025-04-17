import os
import pickle
import pandas as pd
import shap
import streamlit as st


def get_shap_results(sentiment_pipeline):
    # Check if sentiment results already exist
    integrated_data_file = "data/integrated_data.csv"
    shap_coursecontent_file = "models/coursecontent_shap_values.pkl"
    shap_labwork_file = "models/labwork_shap_values.pkl"

    if os.path.exists(shap_coursecontent_file) and os.path.exists(shap_labwork_file):
        print("Loading cached shap results...")
        with open(shap_coursecontent_file, "rb") as f:
            coursecontent_shap_values = pickle.load(f)
        with open(shap_labwork_file, "rb") as f:
            labwork_shap_values = pickle.load(f)
    else:
        # Load integrated data
        data = pd.read_csv(integrated_data_file)

        # Use SHAP to explain BERT predictions
        print("Computing SHAP explanations...")
        explainer = shap.Explainer(sentiment_pipeline)

        # Select a subset of data for SHAP explanation (first 10 rows)
        subset_data = data[["coursecontent_text", "labwork_text"]].head(10)
        coursecontent_texts_subset = subset_data["coursecontent_text"].fillna("").tolist()
        labwork_texts_subset = subset_data["labwork_text"].fillna("").tolist()

        # Compute SHAP values
        coursecontent_shap_values = explainer(coursecontent_texts_subset)

        labwork_shap_values = explainer(labwork_texts_subset)

        # Save the results to cache
        with open(shap_coursecontent_file, "wb") as f:
            pickle.dump(coursecontent_shap_values, f)
        with open(shap_labwork_file, "wb") as f:
            pickle.dump(labwork_shap_values, f)

        # # Save results to session state for use in shap_explanations.py page
        # st.session_state["coursecontent_shap_values"] = coursecontent_shap_values
        # st.session_state["labwork_shap_values"] = labwork_shap_values

    print("SHAP Explanations loaded successfully.")
    return coursecontent_shap_values, labwork_shap_values
