import streamlit as st
import numpy as np
import os

from utils.preprocess import preprocess_data
from scripts.modeling import train_and_evaluate_models
from components.decision_making_unsupervised import unsupervised_decision_making
from components.decision_making_supervised import supervised_decision_making
from components.fusion_adaptation import fusion_and_adaptation
from components.action_execution import action_execution
from components.monitoring_feedback import MonitoringFeedback


def page():
    st.title("Overview")
    st.write("""This application integrates sentiment analysis with learning analytics to provide insights into 
    student performance and engagement. It uses two learning paths:
    """)
    st.markdown("""
    - **Unsupervised Learning Path**: Includes anomaly detection and clustering to identify patterns and outliers.
    - **Supervised Learning Path**: Predicts student dropout risk, performance, engagement, and identifies student groups.
    """)
    st.write("""
        The system then makes decisions, executes actions, and monitors outcomes to improve student success.
    """)

    # Check if data is uploaded
    if not (st.session_state.get('has_analytics', False) and st.session_state.get('has_feedback', False)):
        st.warning("Please upload both Analytics Data and Feedback Data using the Data Upload page before proceeding.")
        return

    # Retrieve data from session state
    analytics_df = st.session_state['analytics_data']
    feedback_df = st.session_state['feedback_data']

    # Preprocess data
    with st.spinner("Preprocessing data..."):
        merged_df, reduced_integrated, reduced_analytics, integrated_features, analytics_features, X_integrated = preprocess_data(
            analytics_df, feedback_df
        )

    # Train and evaluate models (Supervised Learning Path with dropout, performance, engagement predictions)
    with st.spinner("Training models..."):
        (dropout_model, performance_model, engagement_model, groups,
         integrated_acc, analytics_acc, integrated_report, analytics_report,
         integrated_cv, analytics_cv, merged_df, test_idx) = train_and_evaluate_models(reduced_integrated, reduced_analytics,
                                                                                       merged_df)

    # Update merged_df with predictions from train_and_evaluate_models
    merged_df["dropout_pred_int"] = merged_df["dropout_pred_int"]
    merged_df["dropout_pred_ana"] = merged_df["dropout_pred_ana"]
    merged_df["performance_pred"] = merged_df["performance_pred"]
    merged_df["engagement_pred"] = merged_df["engagement_pred"]

    # Unsupervised Decision Making
    with st.spinner("Performing Unsupervised Decision Making..."):
        # unsupervised_model_path = "models/unsupervised"
        # Run on the full dataset to save the models
        full_decisions = unsupervised_decision_making(X_integrated, student_id=None, save_models=True)
        st.session_state["unsupervised_decisions"] = full_decisions
        st.session_state["unsupervised_models"] = full_decisions.get("models", {})  # Store pretrained models

        # Extract decisions for test students only, using Student_ID
        unsupervised_decisions_list = []
        test_student_ids = merged_df.iloc[test_idx]["Student_ID"].unique() if "Student_ID" in merged_df.columns else merged_df.index[test_idx]
        for student_id in test_student_ids:
            # Find the index corresponding to this Student_ID
            idx = merged_df[merged_df["Student_ID"] == student_id].index[0] if "Student_ID" in merged_df.columns else student_id
            cluster_label = full_decisions["cluster"][idx] if isinstance(full_decisions["cluster"], np.ndarray) else full_decisions["cluster"]
            is_anomaly = full_decisions["is_anomaly"][idx] if isinstance(full_decisions["is_anomaly"], np.ndarray) else full_decisions["is_anomaly"]
            teaching_strategy = full_decisions["teaching_strategy"][idx] if isinstance(
                full_decisions["teaching_strategy"], list) else full_decisions["teaching_strategy"]
            review_flag = f"Student {student_id} flagged for review due to anomalous behavior." if is_anomaly else None
            unsupervised_decisions_list.append({
                "student_id": student_id,
                "cluster": cluster_label,
                "is_anomaly": is_anomaly,
                "teaching_strategy": teaching_strategy,
                "review_flag": review_flag
            })
            # Log using Student_ID
            # st.write(f"Processed unsupervised decisions for Student ID: {student_id}, Cluster: {cluster_label}, Anomaly: {is_anomaly}")

    # Supervised Decision Making
    with st.spinner("Performing Supervised Decision Making..."):
        # Process each test student for supervised decisions
        supervised_decisions_list = []
        test_student_ids = merged_df.iloc[test_idx]["Student_ID"].unique() if "Student_ID" in merged_df.columns else merged_df.index[test_idx]
        for student_id in test_student_ids:
            # Filter student data by Student_ID
            student_data = merged_df[merged_df["Student_ID"] == student_id] if "Student_ID" in merged_df.columns else merged_df.loc[[student_id]]
            coursecontent_text = student_data["coursecontent_text"].iloc[0] if "coursecontent_text" in student_data.columns else ""
            labwork_text = student_data["labwork_text"].iloc[0] if "labwork_text" in student_data.columns else ""
            decisions = supervised_decision_making(student_data, coursecontent_text, labwork_text, student_id,
                                                   model_type="integrated")
            supervised_decisions_list.append(decisions)
            # Log using Student_ID
            # st.write(f"Processed supervised decisions for Student ID: {student_id}")

    # Extract predictions for display
    performance_pred = merged_df["performance_pred"].values
    engagement_pred = merged_df["engagement_pred"].values
    # Outliers (from unsupervised decisions)
    outliers = np.array([decision["is_anomaly"] for decision in unsupervised_decisions_list])

    # Display Predictions
    st.subheader("Predictive Modeling Results")
    st.write("**Dropout Risk Predictions (Integrated):**")
    st.write(merged_df.loc[test_idx, ["dropout", "dropout_pred_int"]].head())
    st.write("**Performance Predictions:**")
    st.write(merged_df.loc[test_idx, ["Total_Score", "performance_pred"]].head())
    st.write("**Engagement Predictions:**")
    st.write(merged_df.loc[test_idx, ["engagement_label", "engagement_pred"]].head())

    # Extract actions for display
    unsupervised_actions = []
    for decision in unsupervised_decisions_list:
        if decision["teaching_strategy"]:
            unsupervised_actions.append(
                f"Teaching Strategy for Student {decision['student_id']}: {decision['teaching_strategy']}")
        if decision["review_flag"]:
            unsupervised_actions.append(decision["review_flag"])

    supervised_actions = []
    for decision in supervised_decisions_list:
        supervised_actions.append(decision["feedback"])
        for intervention in decision["interventions"]:
            supervised_actions.append(f"Intervention for Student {decision['student_id']}: {intervention}")

    # Display Actions
    st.subheader("Recommended Actions")
    st.write("**Unsupervised Actions:**")
    for action in unsupervised_actions[:5]:  # Show top 5 for brevity
        st.write(action)
    st.write("**Supervised Actions:**")
    for action in supervised_actions[:5]:  # Show top 5 for brevity
        st.write(action)

    # Save results to session state for use in other pages
    st.session_state["merged_df"] = merged_df
    st.session_state["test_idx"] = test_idx
    st.session_state["clusters"] = [decision["cluster"] for decision in unsupervised_decisions_list]
    st.session_state["anomalies"] = outliers
    st.session_state["performance_pred"] = performance_pred
    st.session_state["engagement_pred"] = engagement_pred
    st.session_state["groups"] = groups
    st.session_state["outliers"] = outliers
    st.session_state["unsupervised_actions"] = unsupervised_actions
    st.session_state["supervised_actions"] = supervised_actions

    # Save comparison results for the Predictions page
    st.session_state["integrated_acc"] = integrated_acc
    st.session_state["analytics_acc"] = analytics_acc
    st.session_state["integrated_report"] = integrated_report
    st.session_state["analytics_report"] = analytics_report
    st.session_state["integrated_cv"] = integrated_cv
    st.session_state["analytics_cv"] = analytics_cv

    # Fusion and Adaptation
    with st.spinner("Fusing decisions..."):
        fused_decisions_list = []
        for unsup_dec, sup_dec in zip(unsupervised_decisions_list, supervised_decisions_list):
            fused_decisions = fusion_and_adaptation(unsup_dec, sup_dec)
            fused_decisions_list.append(fused_decisions)

    # Action Execution
    with st.spinner("Executing actions..."):
        executed_actions_list = []
        for fused_decisions in fused_decisions_list:
            student_id = fused_decisions["student_id"]
            executed_actions = action_execution(fused_decisions, student_id)
            executed_actions_list.append(executed_actions)

    # Monitoring and Feedback Loop
    with st.spinner("Logging actions and refining models..."):
        monitoring = MonitoringFeedback()
        for executed_actions in executed_actions_list:
            student_id = executed_actions["student_id"]
            monitoring.log_actions_and_outcomes(student_id, executed_actions, outcome="Pending", feedback_score=None)
        # Optionally retrain models using the full dataset
        updated_models = monitoring.use_feedback_to_refine_models(X_integrated, reduced_analytics, merged_df)

    # Store results in session state for other pages
    st.session_state["unsupervised_decisions_list"] = unsupervised_decisions_list
    st.session_state["supervised_decisions_list"] = supervised_decisions_list
    st.session_state['executed_actions'] = executed_actions_list
    st.session_state['reduced_integrated'] = reduced_integrated
    st.session_state['X_integrated'] = X_integrated

    st.info("Explore more details in the other pages under 'Student Insights' and 'Analysis Tools'.")


page()