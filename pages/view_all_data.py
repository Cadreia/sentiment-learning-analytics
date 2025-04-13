# pages/view_all_data.py
import streamlit as st
import pandas as pd

def page():
    st.title("View All Data")

    # Check if data is available
    if 'integrated_data' not in st.session_state:
        st.warning("No integrated data available. Please run the analysis on the Overview page first.")
        return

    # Retrieve integrated data
    data = st.session_state['integrated_data']

    st.write("Below is the integrated dataset combining analytics and feedback data, along with predictions and actions.")
    st.dataframe(data)

page()