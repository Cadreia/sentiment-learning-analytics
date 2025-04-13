# pages/overview.py
import streamlit as st
from utils.preprocess import preprocess_data
from models.sentiment_analysis import load_bert_model
from models.predictive_modeling import train_and_evaluate_models
from components.learning_path_unsupervised import learning_path_unsupervised
from components.learning_path_supervised import learning_path_supervised
from components.decision_making_unsupervised import decision_making_unsupervised
from components.decision_making_supervised import decision_making_supervised
from components.fusion_adaptation import fusion_adaptation
from components.decision_fusion import decision_fusion
from components.action_execution import action_execution
from components.monitoring import monitoring_and_feedback
from scripts.actions import run_actions

def page():
    st.title("Overview")
    st.write("""
    This application integrates sentiment analysis with learning analytics to provide insights into student performance and engagement.
    It uses two learning paths:
    - **Unsupervised Learning Path**: Includes anomaly detection and clustering to identify patterns and outliers.
    - **Supervised Learning Path**: Predicts student performance, engagement, groups, and detects outliers.
    
    The system then makes decisions, executes actions, and monitors outcomes to improve student success.
    """)

    # Check if data is uploaded
    if not (st.session_state.get('has_analytics', False) and st.session_state.get('has_feedback', False)):
        st.warning("Please upload both Analytics Data and Feedback Data using the Data Upload page before proceeding.")
        return

    # Retrieve data from session state
    analytics_df = st.session_state['analytics_data']
    feedback_df = st.session_state['feedback_data']

    # Load BERT model
    with st.spinner("Loading BERT model..."):
        tokenizer, model = load_bert_model()

    # Preprocess data
    with st.spinner("Preprocessing data..."):
        merged_df, reduced_features, features = preprocess_data(analytics_df, feedback_df, tokenizer, model)

    # Use dropout label as the target variable
    y = merged_df["dropout"]

    # Train and evaluate models (Supervised Learning Path)
    with st.spinner("Training models..."):
        (rf_integrated, rf_analytics, integrated_acc, analytics_acc,
         integrated_report, analytics_report, integrated_cv, analytics_cv) = train_and_evaluate_models(reduced_features, y)

    # Display comparison results
    st.header("Model Comparison: With vs Without Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("With Sentiment Analysis (Integrated)")
        st.write(f"Accuracy: {integrated_acc:.2f}")
        st.write(f"Cross-Validation Score: {integrated_cv:.2f}")
        st.text("Classification Report:")
        st.text(integrated_report)
    with col2:
        st.subheader("Without Sentiment Analysis (Analytics Only)")
        st.write(f"Accuracy: {analytics_acc:.2f}")
        st.write(f"Cross-Validation Score: {analytics_cv:.2f}")
        st.text("Classification Report:")
        st.text(analytics_report)

    # Unsupervised Learning Path (Anomaly Detection, Clustering)
    with st.spinner("Running Unsupervised Learning Path..."):
        clusters, anomalies = learning_path_unsupervised(reduced_features)

    # Supervised Learning Path (Predict Performance, Groups, Engagement, Outliers)
    with st.spinner("Running Supervised Learning Path..."):
        performance_pred, engagement_pred, groups, outliers = learning_path_supervised(reduced_features, rf_integrated, merged_df)

    # Store additional predictions in merged_df for actions.py
    merged_df["performance_pred"] = performance_pred
    merged_df["engagement_pred"] = engagement_pred
    merged_df["learning_path"] = clusters
    merged_df["anomaly"] = anomalies
    merged_df["outlier"] = outliers

    # Decision Making (Unsupervised)
    with st.spinner("Performing Unsupervised Decision Making..."):
        unsupervised_actions = decision_making_unsupervised(reduced_features, clusters, anomalies)

    # Decision Making (Supervised)
    with st.spinner("Performing Supervised Decision Making..."):
        supervised_actions = decision_making_supervised(reduced_features, rf_integrated, performance_pred, engagement_pred, groups, outliers)

    # Fusion and Adaptation
    with st.spinner("Fusing decisions..."):
        fused_actions = fusion_adaptation(unsupervised_actions, supervised_actions)

    # Decision Fusion
    with st.spinner("Preparing actions..."):
        final_actions = decision_fusion(fused_actions)

    # Action Execution
    with st.spinner("Executing actions..."):
        executed_actions = action_execution(final_actions)

    # Monitoring and Feedback Loop
    with st.spinner("Logging actions and refining models..."):
        executed_actions_with_feedback, feedback_result = monitoring_and_feedback(executed_actions, rf_integrated, reduced_features, y)

    # Run agent actions (for agent_actions.py page)
    with st.spinner("Running agent actions..."):
        integrated_data, executed_actions_with_feedback, feedback_result = run_actions(analytics_df, feedback_df)

    # Store results in session state for other pages
    st.session_state['integrated_data'] = integrated_data
    st.session_state['executed_actions'] = executed_actions_with_feedback
    st.session_state['feedback_result'] = feedback_result
    st.session_state['reduced_features'] = reduced_features
    st.session_state['clusters'] = clusters
    st.session_state['model'] = rf_integrated

    # Display actions on the overview page
    st.header("Action Execution")
    for action in executed_actions_with_feedback:
        st.write(action)

    # Display feedback logs
    st.header("Feedback Logs")
    for action in executed_actions_with_feedback:
        if "Feedback for Student" in action:
            st.write(action)

    st.info("Explore more details in the other pages under 'Student Insights' and 'Analysis Tools'.")

page()