import streamlit as st
import pandas as pd


def page():
    st.title("Agent Actions")
    st.header("Agent Module Actions")

    # Check if actions have been executed and data is available in session state
    if 'executed_actions' not in st.session_state or 'merged_df' not in st.session_state:
        st.warning("No actions have been executed yet. Please run the analysis on the Overview page first.")
        return

    st.write("This section displays the actions taken by the Agent Module based on clustering and engagement "
             "predictions.")

    # Retrieve integrated data
    data = st.session_state['merged_df']

    # Display flagged students
    st.subheader("Flagged Students for Review")
    executed_actions = st.session_state['executed_actions']
    flagged_data = []
    for action in executed_actions:
        student_id = action["student_id"]
        if action["review_flag"]:
            student_data = data.loc[student_id]
            flagged_data.append({
                "Student_ID": student_id,
                "engagement_pred": student_data["engagement_pred"],
                "dropout_pred": student_data["dropout_pred_int"],
                "performance_pred": student_data["performance_pred"],
                "action": action["review_flag"]
            })
    flagged_students_df = pd.DataFrame(flagged_data)
    if not flagged_students_df.empty:
        st.dataframe(flagged_students_df)
    else:
        st.write("No students flagged for review.")

    # Display teaching strategies and interventions
    st.subheader("Teaching Strategies and Interventions")
    strategy_intervention_data = []
    for idx, action in enumerate(executed_actions):
        student_id = action["student_id"]
        student_data = data.loc[student_id]
        teaching_strategy = action["content_adjustments"] if action[
            "content_adjustments"] else "Maintain current strategy"
        intervention = ", ".join(action["triggered_interventions"]) if action["triggered_interventions"] else "None"
        strategy_intervention_data.append({
            "Student_ID": student_id,
            "engagement_pred": student_data["engagement_pred"],
            "dropout_pred": student_data["dropout_pred_int"],
            "performance_pred": student_data["performance_pred"],
            "teaching_strategy": teaching_strategy,
            "intervention": intervention
        })
    strategy_intervention_df = pd.DataFrame(strategy_intervention_data)
    st.dataframe(strategy_intervention_df)

    # Display feedback and monitoring logs
    st.subheader("Feedback Log")
    try:
        action_log = pd.read_csv("data/action_log.csv")
        feedback_log = action_log[action_log["action_type"] == "message"]
        st.dataframe(feedback_log)
    except FileNotFoundError:
        st.warning("Action log not found. Please ensure actions have been executed successfully.")

    st.subheader("Monitoring Log")
    try:
        action_log = pd.read_csv("data/action_log.csv")
        monitoring_log = action_log[action_log["action_type"].isin(["content_adjustment", "intervention"])]
        st.dataframe(monitoring_log)
    except FileNotFoundError:
        st.warning("Action log not found. Please ensure actions have been executed successfully.")


page()
