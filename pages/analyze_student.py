# pages/analyze_student.py
import streamlit as st
import pandas as pd

def page():
    st.title("Analyze Student")

    # Check if data is available
    if 'integrated_data' not in st.session_state:
        st.warning("No integrated data available. Please run the analysis on the Overview page first.")
        return

    # Retrieve integrated data
    data = st.session_state['integrated_data']

    # Select a student by ID
    student_ids = data['Student_ID'].astype(str).tolist()
    selected_student = st.selectbox("Select a Student ID:", student_ids)

    # Filter data for the selected student
    student_data = data[data['Student_ID'].astype(str) == selected_student]

    if student_data.empty:
        st.error("No data found for the selected student.")
        return

    st.subheader(f"Analysis for Student ID: {selected_student}")
    st.write("**Student Details:**")
    st.write(student_data[['First_Name', 'Last_Name', 'Department', 'Grade', 'Total_Score']])

    st.write("**Predictions and Insights:**")
    st.write(student_data[['dropout_pred', 'engagement_pred', 'performance_pred', 'learning_path', 'anomaly', 'outlier']])

    st.write("**Actions Taken:**")
    executed_actions = st.session_state.get('executed_actions', [])
    student_actions = [action for action in executed_actions if f"Student {data.index[data['Student_ID'].astype(str) == selected_student][0]}" in action]
    for action in student_actions:
        st.write(action)

page()