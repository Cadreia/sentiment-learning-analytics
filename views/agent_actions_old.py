# pages/agent_actions.py
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
    flagged_students = data[data["action"].str.contains("Flag Student", na=False)][
        ["Student_ID", "learning_path", "engagement_pred", "dropout_pred", "performance_pred", "action"]]
    st.dataframe(flagged_students)

    # Display teaching strategies and interventions
    st.subheader("Teaching Strategies and Interventions")
    executed_actions = st.session_state['executed_actions']
    teaching_strategies = []
    interventions = []
    for action in executed_actions:
        if "Adapt Teaching Strategy" in action:
            teaching_strategies.append(action)
        elif "Suggest Intervention" in action or "Trigger Intervention" in action:
            interventions.append(action)

    strategy_intervention_df = pd.DataFrame({
        "Student_ID": data["Student_ID"],
        "learning_path": data["learning_path"],
        "engagement_pred": data["engagement_pred"],
        "dropout_pred": data["dropout_pred"],
        "performance_pred": data["performance_pred"],
        "teaching_strategy": [next((s for s in teaching_strategies if f"Student {i}" in s), "Maintain current strategy")
                              for i in range(len(data))],
        "intervention": [next((i for i in interventions if f"Student {idx}" in i), "None") for idx in range(len(data))]
    })
    st.dataframe(strategy_intervention_df)

    # Display feedback and monitoring logs
    st.subheader("Feedback Log")
    try:
        feedback_log = pd.read_csv("data/feedback_log.csv")
        st.dataframe(feedback_log)
    except FileNotFoundError:
        st.warning("Feedback log not found. Please ensure actions have been executed successfully.")

    st.subheader("Monitoring Log")
    try:
        monitoring_log = pd.read_csv("data/monitoring_log.csv")
        st.dataframe(monitoring_log)
    except FileNotFoundError:
        st.warning("Monitoring log not found. Please ensure actions have been executed successfully.")


page()
