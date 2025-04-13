# utils/preprocess.py
import pandas as pd
from utils.data_preprocessing import preprocess_data as preprocess_data_core
import streamlit as st


def preprocess_data(analytics_data, feedback_data, tokenizer, model):
    try:
        integrated_data, reduced_features, features = preprocess_data_core(analytics_data, feedback_data, tokenizer,
                                                                           model)
        integrated_data["dropout"] = integrated_data["Grade"].apply(lambda x: 1 if x == "F" else 0)
        return integrated_data, reduced_features, features
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        raise
