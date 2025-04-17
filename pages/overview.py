import streamlit as st
from utils.preprocess import preprocess_data
from scripts.modeling import train_and_evaluate_models
from components.learning_path_unsupervised import learning_path_unsupervised
from components.learning_path_supervised import learning_path_supervised
from components.decision_making_unsupervised import decision_making_unsupervised
from components.decision_making_supervised import decision_making_supervised
from components.fusion_adaptation import fusion_adaptation
from components.decision_fusion import decision_fusion
from components.action_execution import action_execution


def page():
    st.title("Overview")
    st.write("""
    This application integrates sentiment analysis with learning analytics to provide insights into student performance and engagement.
    It uses two learning paths:
    - **Unsupervised Learning Path**: Includes anomaly detection and clustering to identify patterns and outliers.
    - **Supervised Learning Path**: Predicts student dropout risk, performance, engagement, and identifies student groups.
    
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
         integrated_cv, analytics_cv, merged_df) = train_and_evaluate_models(reduced_integrated, reduced_analytics,
                                                                             merged_df)

    # Update merged_df with predictions from train_and_evaluate_models
    # Since train_and_evaluate_models modifies its 'data' parameter (a copy of merged_df), we copy the predictions back
    merged_df["dropout_pred_int"] = merged_df["dropout_pred_int"]
    merged_df["dropout_pred_ana"] = merged_df["dropout_pred_ana"]
    merged_df["performance_pred"] = merged_df["performance_pred"]
    merged_df["engagement_pred"] = merged_df["engagement_pred"]

    # Unsupervised Learning Path (Clustering and Anomaly Detection)
    with st.spinner("Running Unsupervised Learning Path..."):
        clusters, anomalies = learning_path_unsupervised(reduced_integrated)

    # Supervised Learning Path (Using the predictions from train_and_evaluate_models)
    with st.spinner("Running Supervised Learning Path..."):
        performance_pred, engagement_pred, groups, outliers = learning_path_supervised(
            reduced_integrated, groups, merged_df
        )

    # Decision Making (Unsupervised)
    with st.spinner("Performing Unsupervised Decision Making..."):
        unsupervised_actions = decision_making_unsupervised(reduced_integrated, clusters, anomalies)

    # Decision Making (Supervised)
    with st.spinner("Performing Supervised Decision Making..."):
        supervised_actions = decision_making_supervised(
            reduced_integrated, dropout_model, performance_pred, engagement_pred, groups, outliers
        )

    # Display Predictions
    st.subheader("Predictive Modeling Results")
    st.write("**Dropout Risk Predictions (Integrated):**")
    st.write(merged_df[["dropout", "dropout_pred_int"]].head())
    st.write("**Performance Predictions:**")
    st.write(merged_df[["Total_Score", "performance_pred"]].head())
    st.write("**Engagement Predictions:**")
    st.write(merged_df[["engagement_label", "engagement_pred"]].head())

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
    st.session_state["clusters"] = clusters
    st.session_state["anomalies"] = anomalies
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

    # # Store additional predictions in merged_df for actions.py
    # merged_df["dropout_pred"] = merged_df["dropout_pred"].apply(lambda x: "High" if x == 1 else "Low")
    # merged_df["performance_pred"] = performance_pred
    # merged_df["engagement_pred"] = engagement_pred
    # merged_df["learning_path"] = groups
    # merged_df["anomaly"] = anomalies
    # merged_df["outlier"] = outliers
    #
    # Fusion and Adaptation
    with st.spinner("Fusing decisions..."):
        fused_actions = fusion_adaptation(unsupervised_actions, supervised_actions)

    # Decision Fusion
    with st.spinner("Preparing actions..."):
        final_actions = decision_fusion(fused_actions)

    # Action Execution
    with st.spinner("Executing actions..."):
        executed_actions = action_execution(final_actions)

    # # Monitoring and Feedback Loop
    # with st.spinner("Logging actions and refining models..."):
    #     executed_actions_with_feedback, feedback_result = monitoring_and_feedback(executed_actions, dropout_model,
    #                                                                               reduced_integrated, y)
    #
    # # Run agent actions (for agent_actions.py page)
    # with st.spinner("Running agent actions..."):
    #     integrated_data, executed_actions_with_feedback, feedback_result = run_actions(analytics_df, feedback_df)

    # # Store results in session state for other pages
    # st.session_state['integrated_data'] = integrated_data
    # st.session_state['executed_actions'] = executed_actions_with_feedback
    # st.session_state['feedback_result'] = feedback_result
    # st.session_state['model'] = dropout_model  # For SHAP explanations
    #
    # # Display actions on the overview page
    # st.header("Action Execution")
    # for action in executed_actions_with_feedback:
    #     st.write(action)
    #
    # # Display feedback logs
    # st.header("Feedback Logs")
    # for action in executed_actions_with_feedback:
    #     if "Feedback for Student" in action:
    #         st.write(action)

    st.session_state['reduced_integrated'] = reduced_integrated
    st.session_state['X_integrated'] = X_integrated

    st.info("Explore more details in the other pages under 'Student Insights' and 'Analysis Tools'.")


page()
