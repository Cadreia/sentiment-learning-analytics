import streamlit as st
import pandas as pd
from pathlib import Path

def page():
    # Define data directory and ensure it exists
    DATA_DIR = Path("./data")
    DATA_DIR.mkdir(exist_ok=True)
    ANALYTICS_PATH = DATA_DIR / "student_assessments.csv"
    FEEDBACK_PATH = DATA_DIR / "student_feedback.csv"

    # Initialize session state variables only if they don't exist
    if 'has_analytics' not in st.session_state:
        st.session_state['has_analytics'] = False
    if 'has_feedback' not in st.session_state:
        st.session_state['has_feedback'] = False
    if 'analytics_data' not in st.session_state:
        st.session_state['analytics_data'] = None
    if 'feedback_data' not in st.session_state:
        st.session_state['feedback_data'] = None

    # Check for existing files on disk and load them
    def load_existing_data():
        # Load analytics data if file exists
        if ANALYTICS_PATH.exists():
            try:
                analytics_data = pd.read_csv(ANALYTICS_PATH)
                expected_analytics_columns = {
                    'Student_ID', 'First_Name', 'Last_Name', 'Email', 'Gender', 'Age', 'Department',
                    'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg',
                    'Participation_Score', 'Projects_Score', 'Total_Score', 'Grade', 'Study_Hours_per_Week',
                    'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level',
                    'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
                }
                if set(analytics_data.columns) == expected_analytics_columns:
                    st.session_state['analytics_data'] = analytics_data
                    st.session_state['has_analytics'] = True
                else:
                    st.warning("Existing Analytics Data has incorrect columns. Please re-upload.")
            except Exception as e:
                st.warning(f"Error loading existing Analytics Data: {str(e)}")

        # Load feedback data if file exists
        if FEEDBACK_PATH.exists():
            try:
                feedback_data = pd.read_csv(FEEDBACK_PATH)
                expected_feedback_columns = {
                    'teaching', 'teaching.1', 'coursecontent', 'coursecontent.1', 'examination',
                    'Examination', 'labwork', 'labwork.1', 'library_facilities', 'library_facilities.1',
                    'extracurricular', 'extracurricular.1'
                }
                if set(feedback_data.columns) == expected_feedback_columns:
                    st.session_state['feedback_data'] = feedback_data
                    st.session_state['has_feedback'] = True
                else:
                    st.warning("Existing Feedback Data has incorrect columns. Please re-upload.")
            except Exception as e:
                st.warning(f"Error loading existing Feedback Data: {str(e)}")

    # Load existing data on page load
    load_existing_data()

    st.title("Data Upload")
    st.markdown("""
    Upload your student data here to analyze it with the app. You must upload **both** the analytics data and feedback data files. 
    Each file must be a **CSV** and follow the exact format below. Incorrect formats will be rejected.
    If data is already loaded from a previous session, you can choose to keep it or upload new files to replace it.
    """)

    # Define expected column formats
    st.header("Expected File Formats")
    st.markdown("""
    - **Analytics Data (student_assessments.csv)**: Must contain these exact columns:
        - Student_ID (integer): Unique identifier for each student.
        - First_Name (string): Student’s first name.
        - Last_Name (string): Student’s last name.
        - Email (string): Contact email (can be anonymized).
        - Gender (string): Male, Female, Other.
        - Age (integer): The age of the student.
        - Department (string): Student's department (e.g., CS, Engineering, Business).
        - Attendance (%) (float/integer): Attendance percentage (0-100).
        - Midterm_Score (float/integer): Midterm exam score (out of 100).
        - Final_Score (float/integer): Final exam score (out of 100).
        - Assignments_Avg (float/integer): Average score of all assignments (out of 100).
        - Quizzes_Avg (float/integer): Average quiz scores (out of 100).
        - Participation_Score (float/integer): Score based on class participation (0-10).
        - Projects_Score (float/integer): Project evaluation score (out of 100).
        - Total_Score (float/integer): Weighted sum of all grades.
        - Grade (string): Letter grade (A, B, C, D, F).
        - Study_Hours_per_Week (float/integer): Average study hours per week.
        - Extracurricular_Activities (string): Whether the student participates in extracurriculars (Yes/No).
        - Internet_Access_at_Home (string): Does the student have access to the internet at home? (Yes/No).
        - Parent_Education_Level (string): Highest education level of parents (None, High School, Bachelor's, Master's, PhD).
        - Family_Income_Level (string): Low, Medium, High.
        - Stress_Level (1-10) (integer): Self-reported stress level (1: Low, 10: High).
        - Sleep_Hours_per_Night (float/integer): Average hours of sleep per night.

    - **Feedback Data (student_feedback.csv)**: Must contain these exact columns:
        - teaching (string): Feedback on teaching.
        - teaching.1 (string): Sentiment label for teaching feedback (e.g., Positive, Negative, Neutral).
        - coursecontent (string): Feedback on course content.
        - coursecontent.1 (string): Sentiment label for course content feedback.
        - examination (string): Feedback on examination.
        - Examination (string): Sentiment label for examination feedback.
        - labwork (string): Feedback on lab work.
        - labwork.1 (string): Sentiment label for lab work feedback.
        - library_facilities (string): Feedback on library facilities.
        - library_facilities.1 (string): Sentiment label for library facilities feedback.
        - extracurricular (string): Feedback on extracurricular activities.
        - extracurricular.1 (string): Sentiment label for extracurricular feedback.
    """)

    # File uploaders for both files
    st.header("Upload New Data")
    st.markdown("Upload files only if you want to replace existing data. Current data will be kept unless overwritten.")
    col1, col2 = st.columns([3, 1])
    with col1:
        analytics_file = st.file_uploader("Upload Analytics Data (student_assessments.csv)", type="csv")
        feedback_file = st.file_uploader("Upload Feedback Data (student_feedback.csv)", type="csv")
    with col2:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button("Clear All Data"):
            if ANALYTICS_PATH.exists():
                ANALYTICS_PATH.unlink()
            if FEEDBACK_PATH.exists():
                FEEDBACK_PATH.unlink()
            st.session_state['has_analytics'] = False
            st.session_state['has_feedback'] = False
            st.session_state['analytics_data'] = None
            st.session_state['feedback_data'] = None
            if 'merged_df' in st.session_state:
                del st.session_state['merged_df']  # Clear merged_df to avoid stale data
            st.success("All data has been cleared from both disk and session.")
            st.rerun()

    # Process Analytics Data
    if analytics_file is not None:
        try:
            analytics_data = pd.read_csv(analytics_file)
            expected_analytics_columns = {
                'Student_ID', 'First_Name', 'Last_Name', 'Email', 'Gender', 'Age', 'Department',
                'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg',
                'Participation_Score', 'Projects_Score', 'Total_Score', 'Grade', 'Study_Hours_per_Week',
                'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level',
                'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
            }
            actual_columns = set(analytics_data.columns)
            missing_columns = expected_analytics_columns - actual_columns
            extra_columns = actual_columns - expected_analytics_columns

            if missing_columns:
                st.error(f"Missing required columns in Analytics Data: {', '.join(missing_columns)}.")
            elif extra_columns:
                st.error(f"Extra columns found in Analytics Data: {', '.join(extra_columns)}.")
            else:
                # Save to disk
                analytics_data.to_csv(ANALYTICS_PATH, index=False)
                st.session_state['analytics_data'] = analytics_data
                st.session_state['has_analytics'] = True
                if 'merged_df' in st.session_state:
                    del st.session_state['merged_df']  # Invalidate merged_df on new upload
                st.success("Analytics Data uploaded and saved successfully!")

        except Exception as e:
            st.error(f"Error processing the Analytics Data file: {str(e)}.")

    # Process Feedback Data
    if feedback_file is not None:
        try:
            feedback_data = pd.read_csv(feedback_file)
            expected_feedback_columns = {
                'teaching', 'teaching.1', 'coursecontent', 'coursecontent.1', 'examination',
                'Examination', 'labwork', 'labwork.1', 'library_facilities', 'library_facilities.1',
                'extracurricular', 'extracurricular.1'
            }
            actual_columns = set(feedback_data.columns)
            missing_columns = expected_feedback_columns - actual_columns
            extra_columns = actual_columns - expected_feedback_columns

            if missing_columns:
                st.error(f"Missing required columns in Feedback Data: {', '.join(missing_columns)}.")
            elif extra_columns:
                st.error(f"Extra columns found in Feedback Data: {', '.join(extra_columns)}.")
            else:
                # Save to disk
                feedback_data.to_csv(FEEDBACK_PATH, index=False)
                st.session_state['feedback_data'] = feedback_data
                st.session_state['has_feedback'] = True
                if 'merged_df' in st.session_state:
                    del st.session_state['merged_df']  # Invalidate merged_df on new upload
                st.success("Feedback Data uploaded and saved successfully!")

        except Exception as e:
            st.error(f"Error processing the Feedback Data file: {str(e)}.")

    # Show current data status
    st.header("Current Data Status")
    if st.session_state.get('has_analytics', False) and st.session_state['analytics_data'] is not None:
        st.markdown("- **Analytics Data**: Loaded")
        st.markdown("**Preview:**")
        st.dataframe(st.session_state['analytics_data'].head())
    else:
        st.markdown("- **Analytics Data**: Not Loaded")

    if st.session_state.get('has_feedback', False) and st.session_state['feedback_data'] is not None:
        st.markdown("- **Feedback Data**: Loaded")
        st.markdown("**Preview:**")
        st.dataframe(st.session_state['feedback_data'].head())
    else:
        st.markdown("- **Feedback Data**: Not Loaded")

    if not (st.session_state.get('has_analytics', False) and st.session_state.get('has_feedback', False)):
        st.warning("Please upload both Analytics Data and Feedback Data to proceed with the analysis.")
    else:
        st.info("Both files are loaded successfully! You can proceed with the analysis in the main app.")

page()