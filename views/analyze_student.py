import streamlit as st
import pandas as pd
from scripts.sentiment_analysis import get_sentiment_scores, load_sentiment_pipeline
from scripts.predictions import predict, recommend_action
from utils.data_preprocessing import prepare_student_features
from components.decision_making_unsupervised import unsupervised_decision_making
from components.decision_making_supervised import supervised_decision_making
from components.fusion_adaptation import fusion_and_adaptation
from components.action_execution import action_execution
from components.monitoring_feedback import MonitoringFeedback  # Added import

# Load sentiment pipeline once at module level
sentiment_pipeline = load_sentiment_pipeline()

# Custom CSS for professional styling (aligned with view_all_predictions.py)
st.markdown("""
    <style>
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1e88e5;
        margin-bottom: 10px;
    }
     .sub-section {
        background-color: white;
        padding: 10px;
        border: 0;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1e88e5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        color: white;
    }
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


def extract_scalar(value, name="value"):
    """Extract a single scalar from a value, handling lists or arrays."""
    if isinstance(value, (list, pd.Series, pd.DataFrame)) and len(value) > 0:
        return value[0] if not isinstance(value[0], (list, pd.Series, pd.DataFrame)) else str(value[0])
    elif isinstance(value, (int, float, str)):
        return value
    else:
        st.warning(f"Unexpected {name} format: {type(value)}. Using default.")
        return str(value)


def analyze_student_page():
    # Main container
    st.markdown('<h1 class="section-header">Analyze Student Performance</h1>', unsafe_allow_html=True)
    st.markdown(
        "Select a student from the test set or enter new student details to predict performance, engagement, dropout risk, and recommended actions.",
        unsafe_allow_html=True)

    # Load cached data
    data = st.session_state.get('merged_df')
    test_idx = st.session_state.get('test_idx')  # Get test set indices
    if data is None or test_idx is None:
        st.error("No cached data or test indices available. Please run the analysis on the Overview page first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Input method selection
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Choose Analysis Method</h2>', unsafe_allow_html=True)
    input_method = st.radio(
        "Select Input Method",
        ["Select Existing Student", "Enter New Student Data"],
        key="input_method",
        horizontal=True
    )

    student_data = None
    coursecontent_text = ""
    labwork_text = ""
    student_id = "New_Student"

    if input_method == "Select Existing Student":
        # Dropdown for student selection (test set only)
        test_student_ids = data.index[test_idx] if "Student_ID" not in data.columns else data.iloc[test_idx]["Student_ID"].unique()
        selected_id = st.selectbox("Select Student ID", options=[""] + list(test_student_ids), key="student_select")

        if selected_id:
            # Retrieve student data based on Student_ID or index
            if "Student_ID" in data.columns:
                student_data = data[data["Student_ID"] == selected_id]  # Filter by Student_ID
                if student_data.empty:
                    st.error(f"No student found with Student_ID: {selected_id}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
            else:
                student_data = data.loc[[selected_id]]  # Use index
            student_id = selected_id
            coursecontent_text = student_data.get("coursecontent_text_cleaned", "No feedback available").iloc[0]
            labwork_text = student_data.get("labwork_text_cleaned", "No feedback available").iloc[0]

            # Display selected student’s data
            st.markdown("**Selected Student Data**")
            summary_df = pd.DataFrame({
                "Student ID": [student_id],
                "Attendance (%)": [student_data["Attendance (%)"].iloc[0] * 100],
                "Total Score": [student_data["Total_Score"].iloc[0]],
                "Course Content Feedback": [coursecontent_text],
                "Lab Work Feedback": [labwork_text]
            })
            st.dataframe(summary_df, use_container_width=True, column_config={"_index": None})
        else:
            st.info("Please select a student to analyze.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return

    else:
        # Form for new student data
        with st.form(key="student_form"):
            st.markdown('<h2 class="section-header">Enter Student Data</h2>', unsafe_allow_html=True)
            student_id = st.text_input("Student ID", "S001", key="student_id")
            col1, col2 = st.columns(2)
            with col1:
                attendance = st.slider("Attendance (%)", 0, 100, 80, key="attendance")
                midterm_score = st.slider("Midterm Score (0-100)", 0, 100, 70, key="midterm")
                final_score = st.slider("Final Score (0-100)", 0, 100, 75, key="final")
                assignments_avg = st.slider("Assignments Average (0-100)", 0, 100, 80, key="assignments")
                quizzes_avg = st.slider("Quizzes Average (0-100)", 0, 100, 85, key="quizzes")
                participation_score = st.slider("Participation Score (0-10)", 0, 10, 8, key="participation")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
                department = st.selectbox("Department", ["Computer Science", "Engineering", "Mathematics", "Other"], key="department")
                grade = st.selectbox("Grade", ["A", "B", "C", "D", "F"], key="grade")
            with col2:
                projects_score = st.slider("Projects Score (0-100)", 0, 100, 90, key="projects")
                total_score = st.slider("Total Score (0-1)", 0.0, 1.0, 0.8, step=0.01, key="total_score")
                study_hours = st.number_input("Study Hours per Week", min_value=0.0, step=0.1, value=10.0, key="study_hours")
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, key="stress")
                sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, step=0.1, value=7.0, key="sleep")
                age = st.number_input("Age", min_value=0, value=20, key="age")
                extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], key="extracurricular")
                internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"], key="internet_access")
                parent_education = st.selectbox("Parent Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"], key="parent_education")
                family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"], key="family_income")
            coursecontent_text = st.text_area("Course Content Feedback", "The course content was well-structured.", key="coursecontent")
            labwork_text = st.text_area("Exercise Content Feedback", "The exercises were challenging but helpful.", key="labwork")
            submit_button = st.form_submit_button(label="Analyze")

        if not submit_button:
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # Create DataFrame for new student
        student_data = pd.DataFrame({
            "Attendance (%)": [attendance],
            "Midterm_Score": [midterm_score],
            "Final_Score": [final_score],
            "Assignments_Avg": [assignments_avg],
            "Quizzes_Avg": [quizzes_avg],
            "Participation_Score": [participation_score],
            "Projects_Score": [projects_score],
            "Total_Score": [total_score],
            "Study_Hours_per_Week": [study_hours],
            "Stress_Level (1-10)": [stress_level],
            "Sleep_Hours_per_Night": [sleep_hours],
            "Age": [age],
            "Gender": [gender],
            "Department": [department],
            "Grade": [grade],
            "Extracurricular_Activities": [extracurricular],
            "Internet_Access_at_Home": [internet_access],
            "Parent_Education_Level": [parent_education],
            "Family_Income_Level": [family_income]
        }, index=[student_id])

    # Perform analysis
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)

    try:
        if input_method == "Select Existing Student":
            # Check for required prediction columns
            required_cols = ["coursecontent_sentiment_score", "labwork_sentiment_score", "dropout_pred_int",
                             "performance_pred", "engagement_pred"]
            missing_cols = [col for col in required_cols if col not in student_data.columns]
            if missing_cols:
                st.error(
                    f"Cached data missing prediction columns: {', '.join(missing_cols)}. Please run the analysis on the Overview page first.")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                return

            # Retrieve cached predictions
            coursecontent_polarity = float(student_data["coursecontent_sentiment_score"].iloc[0])
            labwork_polarity = float(student_data["labwork_sentiment_score"].iloc[0])
            dropout_risk = int(student_data["dropout_pred_int"].iloc[0])
            performance_pred = float(student_data["performance_pred"].iloc[0])
            engagement_pred = str(student_data["engagement_pred"].iloc[0])

            # Retrieve unsupervised decisions from session state
            unsupervised_decisions_list = st.session_state.get("unsupervised_decisions_list", [])
            if not unsupervised_decisions_list:
                st.error("Unsupervised decisions not found. Please run the Overview page first.")
                return
            # Find the decision for the selected student
            unsupervised_decisions = next(
                (decision for decision in unsupervised_decisions_list if decision["student_id"] == student_id),
                None
            )
            if not unsupervised_decisions:
                st.error(f"No unsupervised decisions found for Student ID: {student_id}.")
                return
            cluster_label = unsupervised_decisions["cluster"]
            is_anomaly = unsupervised_decisions["is_anomaly"]
            teaching_strategy = unsupervised_decisions["teaching_strategy"]
            review_flag = unsupervised_decisions["review_flag"]

        else:
            # Prepare student features for new data
            normalized_data, X_integrated = prepare_student_features(
                student_data,
                coursecontent_text,
                labwork_text,
                sentiment_pipeline=sentiment_pipeline
            )
            coursecontent_polarity = normalized_data["coursecontent_sentiment_score"].iloc[0]
            labwork_polarity = normalized_data["labwork_sentiment_score"].iloc[0]

            # Compute predictions for new student
            dropout_risk, _, _ = predict(
                student_data,
                coursecontent_text=coursecontent_text,
                labwork_text=labwork_text,
                model_type="integrated",
                predict_type="dropout"
            )
            performance_pred = predict(
                student_data,
                coursecontent_text=coursecontent_text,
                labwork_text=labwork_text,
                model_type="integrated",
                predict_type="performance"
            )
            engagement_pred = predict(
                student_data,
                coursecontent_text=coursecontent_text,
                labwork_text=labwork_text,
                model_type="integrated",
                predict_type="engagement"
            )

            # Use pretrained unsupervised models
            pre_trained_models = st.session_state.get("unsupervised_models")
            if not pre_trained_models:
                st.error("Unsupervised models not found. Please run the Overview page first.")
                return
            unsupervised_decisions = unsupervised_decision_making(
                X_integrated, student_id, pre_trained_models=pre_trained_models
            )
            # Ensure student_id is included in unsupervised_decisions
            if "student_id" not in unsupervised_decisions:
                unsupervised_decisions["student_id"] = student_id
            cluster_label = unsupervised_decisions["cluster"]
            is_anomaly = unsupervised_decisions["is_anomaly"]
            teaching_strategy = unsupervised_decisions["teaching_strategy"]
            review_flag = unsupervised_decisions["review_flag"]

        # Compute detailed decisions and actions
        supervised_decisions = supervised_decision_making(student_data, coursecontent_text, labwork_text, student_id,
                                                          model_type="integrated")
        fused_decisions = fusion_and_adaptation(unsupervised_decisions, supervised_decisions)
        executed_actions = action_execution(fused_decisions, student_id)
        basic_recommendation = recommend_action(coursecontent_polarity, labwork_polarity,
                                                student_data["Total_Score"].iloc[0])

        # Log actions and refine models for new student data
        if input_method == "Enter New Student Data":
            with st.spinner("Logging actions and refining models..."):
                monitoring = MonitoringFeedback()
                # Log actions for the new student
                monitoring.log_actions_and_outcomes(student_id, executed_actions, outcome="Pending", feedback_score=None)
                # Optionally retrain models using the new student's data
                updated_models = monitoring.use_feedback_to_refine_models(X_integrated, None, student_data)

        # Display predictions (similar to overview.py)
        st.subheader("Predictive Modeling Results")
        st.markdown(f'<div class="metric-box"><b>Course Content Sentiment (BERT):</b> {float(coursecontent_polarity):.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Lab Work Sentiment (BERT):</b> {float(labwork_polarity):.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Dropout Risk:</b> {"High" if int(dropout_risk) == 1 else "Low"}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-box"><b>Predicted Performance (Total Score):</b> {float(performance_pred):.2f}</div>',
            unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Engagement Prediction:</b> {str(engagement_pred)}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Assigned Cluster:</b> {cluster_label}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Anomaly Status:</b> {"Anomaly" if is_anomaly else "Normal"}</div>',
                    unsafe_allow_html=True)

        # Display recommended actions (similar to overview.py)
        st.subheader("Recommended Actions")
        st.markdown("**Unsupervised Actions:**")
        unsupervised_actions = []
        if teaching_strategy:
            unsupervised_actions.append(f"Teaching Strategy for Student {student_id}: {teaching_strategy}")
        if review_flag:
            unsupervised_actions.append(review_flag)
        for action in unsupervised_actions:
            st.markdown(f'<div class="metric-box">{action}</div>', unsafe_allow_html=True)

        st.markdown("**Supervised Actions:**")
        supervised_actions = [supervised_decisions["feedback"]]
        for intervention in supervised_decisions["interventions"]:
            supervised_actions.append(f"Intervention for Student {student_id}: {intervention}")
        for action in supervised_actions:
            st.markdown(f'<div class="metric-box">{action}</div>', unsafe_allow_html=True)

        # Display executed actions
        st.subheader("Executed Actions")
        st.markdown(
            f'<div class="metric-box"><b>Messages Sent:</b>\n{chr(10).join(executed_actions["messages"])}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-box"><b>Content Adjustments:</b> {executed_actions["content_adjustments"]}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-box"><b>Triggered Interventions:</b> {", ".join(executed_actions["triggered_interventions"])}</div>',
            unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}. Please check the input data and try again.")
        st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

analyze_student_page()