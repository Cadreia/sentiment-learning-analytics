import streamlit as st


def page():
    st.title("View All Data")

    # Section 1: Integrated Dataset
    # Check if the analysis has been run
    if "merged_df" not in st.session_state:
        st.error("Please run the analysis on the Overview page first to generate the data.")
        return
    else:
        st.header("Integrated Dataset")
        merged_df = st.session_state["merged_df"]
        st.dataframe(merged_df, use_container_width=True)

    # Section 2: Original Datasets
    st.header("Original Datasets")

    st.subheader("Analytics Data")
    if "analytics_data" in st.session_state:
        analytics_df = st.session_state["analytics_data"]
        st.dataframe(analytics_df, use_container_width=True)
    else:
        st.warning("Analytics data not found in session state.")

    st.subheader("Feedback Data")
    if "feedback_data" in st.session_state:
        feedback_df = st.session_state["feedback_data"]
        st.dataframe(feedback_df, use_container_width=True)
    else:
        st.warning("Feedback data not found in session state.")


page()
