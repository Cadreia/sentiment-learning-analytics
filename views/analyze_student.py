import streamlit as st
import pandas as pd
from scripts.sentiment_analysis import get_sentiment_scores, load_sentiment_pipeline
from scripts.predictions import predict, recommend_action
from utils.data_preprocessing import prepare_student_features
from components.decision_making_unsupervised import unsupervised_decision_making
from components.decision_making_supervised import supervised_decision_making
from components.fusion_adaptation import fusion_and_adaptation
from components.action_execution import action_execution

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
        "Select a student from the cached dataset or enter new student details to predict performance, engagement, dropout risk, and recommended actions.",
        unsafe_allow_html=True)

    # Load cached data
    data = st.session_state.get('merged_df')
    if data is None:
        st.error("No cached data available. Please upload data on the Data Upload page.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # # Define numerical and categorical columns (match preprocess_data and predictions.py)
    # numerical_cols = [
    #     "Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
    #     "Quizzes_Avg", "Participation_Score", "Projects_Score", "Total_Score",
    #     "Study_Hours_per_Week", "Age", "Stress_Level (1-10)", "Sleep_Hours_per_Night"
    # ]
    # categorical_cols = [
    #     "Gender", "Department", "Grade", "Extracurricular_Activities",
    #     "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level"
    # ]

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
        # Dropdown for student selection
        student_ids = data.index if "Student_ID" not in data.columns else data["Student_ID"].unique()
        selected_id = st.selectbox("Select Student ID", options=[""] + list(student_ids), key="student_select")

        if selected_id:
            student_data = data.loc[[selected_id]]  # Keep as DataFrame
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
            with col2:
                projects_score = st.slider("Projects Score (0-100)", 0, 100, 90, key="projects")
                total_score = st.slider("Total Score (0-1)", 0.0, 1.0, 0.8, step=0.01, key="total_score")
                study_hours = st.number_input("Study Hours per Week", min_value=0.0, step=0.1, value=10.0,
                                              key="study_hours")
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, key="stress")
                sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, step=0.1, value=7.0, key="sleep")
                age = st.number_input("Age", min_value=0, value=20, key="age")
            coursecontent_text = st.text_area("Course Content Feedback", "The course content was well-structured.",
                                              key="coursecontent")
            labwork_text = st.text_area("Exercise Content Feedback", "The exercises were challenging but helpful.",
                                        key="labwork")
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
            "Age": [age]
        }, index=[student_id])

    # Perform analysis
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)

    try:
        # # Prepare student features (normalize, compute sentiment, generate embeddings)
        # normalized_data, X_integrated = prepare_student_features(
        #     student_data,
        #     coursecontent_text,
        #     labwork_text,
        #     numerical_cols,
        #     categorical_cols,
        #     sentiment_pipeline=sentiment_pipeline
        # )
        #
        # # Extract sentiment scores from normalized_data
        # coursecontent_polarity = normalized_data["coursecontent_sentiment_score"].iloc[0]
        # labwork_polarity = normalized_data["labwork_sentiment_score"].iloc[0]

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

            # Retrieve cluster from session state
            clusters = st.session_state.get("clusters", [])
            if not clusters:
                st.error("Cluster data not found. Please run the Overview page first.")
                return
            cluster_idx = list(data.index).index(selected_id)
            cluster_label = clusters[cluster_idx]
            # Retrieve anomaly status
            anomalies = st.session_state.get("anomalies", [])
            is_anomaly = anomalies[cluster_idx] if anomalies else False

            # Manually create unsupervised decisions since we already have cluster and anomaly
            teaching_strategy = "Focus on foundational skills and regular check-ins." if cluster_label == 0 else \
                "Encourage advanced projects and peer collaboration." if cluster_label == 1 else \
                    "Provide additional resources and interactive activities."
            review_flag = f"Student {student_id} flagged for review due to anomalous behavior." if is_anomaly else None
            unsupervised_decisions = {
                "student_id": student_id,
                "cluster": cluster_label,
                "is_anomaly": is_anomaly,
                "teaching_strategy": teaching_strategy,
                "review_flag": review_flag
            }

        else:
            # Compute predictions directly
            dropout_risk, normalized_data, X_integrated = predict(
                student_data,
                coursecontent_text=coursecontent_text,
                labwork_text=labwork_text,
                # coursecontent_sentiment=coursecontent_polarity,
                # labwork_sentiment=labwork_polarity,
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

            coursecontent_polarity = normalized_data['coursecontent_sentiment_score']
            labwork_polarity = normalized_data['labwork_sentiment_score']

            # Load pre-trained models for unsupervised decision-making
            pre_trained_models = st.session_state.get("unsupervised_models")
            if not pre_trained_models:
                st.error("Unsupervised models not found. Please run the Overview page first.")
                return

            # Assign to existing cluster using pre-trained models
            unsupervised_decisions = unsupervised_decision_making(X_integrated, student_id,
                                                                  pre_trained_models=pre_trained_models)

        # Compute detailed decisions and actions
        # Supervised decision-making
        supervised_decisions = supervised_decision_making(student_data, coursecontent_text, labwork_text, student_id,
                                                          model_type="integrated")

        # Fusion and adaptation
        fused_decisions = fusion_and_adaptation(unsupervised_decisions, supervised_decisions)

        # Action execution
        executed_actions = action_execution(fused_decisions, student_id)

        # Basic recommendation
        basic_recommendation = recommend_action(coursecontent_polarity, labwork_polarity,
                                                student_data["Total_Score"].iloc[0])

        # Display predictions
        st.markdown(
            f'<div class="metric-box"><b>Course Content Sentiment (BERT):</b> {float(coursecontent_polarity):.2f}</div>',
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

        # Display cluster assignment
        st.markdown(f'<div class="metric-box"><b>Assigned Cluster:</b> {unsupervised_decisions["cluster"]}</div>',
                    unsafe_allow_html=True)

        # Display basic recommendation
        st.markdown("#### Basic Recommendation")
        st.markdown(f'<div class="metric-box"><b>Recommendation:</b> {basic_recommendation}</div>',
                    unsafe_allow_html=True)

        # Display detailed recommendations
        st.markdown("#### Detailed Recommendations")
        feedback = supervised_decisions["feedback"]
        teaching_strategy = unsupervised_decisions["teaching_strategy"]
        interventions = supervised_decisions["interventions"]
        review_flag = unsupervised_decisions["review_flag"]

        st.markdown(f'<div class="metric-box"><b>Feedback:</b>\n{feedback}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Teaching Strategy:</b> {teaching_strategy}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Interventions:</b> {", ".join(interventions)}</div>',
                    unsafe_allow_html=True)
        if review_flag:
            st.markdown(f'<div class="metric-box"><b>Review Flag:</b> {review_flag}</div>', unsafe_allow_html=True)

        # Display executed actions
        st.markdown("#### Executed Actions")
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
