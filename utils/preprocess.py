from utils.data_preprocessing import preprocess_data as preprocess_data_core
import streamlit as st


def preprocess_data(analytics_data, feedback_data):
    try:
        # Call the core preprocessing function
        merged_df, reduced_integrated, reduced_analytics, integrated_features, analytics_features, X_integrated = preprocess_data_core(
            analytics_data, feedback_data
        )

        # Ensure the dropout column is present (already done in preprocess_data_core, but kept here for clarity)
        if "dropout" not in merged_df.columns:
            merged_df["dropout"] = merged_df["Grade"].apply(lambda x: 1 if x == "F" else 0)

        return merged_df, reduced_integrated, reduced_analytics, integrated_features, analytics_features, X_integrated

    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        raise
