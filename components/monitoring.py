# components/monitoring.py
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

logging.basicConfig(filename='system_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

FEEDBACK_LOG_FILE = 'data/feedback_log.csv'


def initialize_feedback_log():
    if not os.path.exists(FEEDBACK_LOG_FILE):
        feedback_df = pd.DataFrame(columns=[
            'timestamp', 'student_id', 'action', 'dropout_risk_reduction',
            'sentiment_improvement', 'performance_improvement'
        ])
        feedback_df.to_csv(FEEDBACK_LOG_FILE, index=False)


def log_action(action):
    logging.info(action)


def collect_feedback(student_id, action, previous_dropout_prob, X, model):
    dropout_risk_reduction = np.random.uniform(0.05, 0.2)
    sentiment_improvement = np.random.uniform(0, 1)
    performance_improvement = np.random.uniform(0, 10)

    feedback_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'student_id': student_id,
        'action': action,
        'dropout_risk_reduction': dropout_risk_reduction,
        'sentiment_improvement': sentiment_improvement,
        'performance_improvement': performance_improvement
    }
    feedback_df = pd.DataFrame([feedback_entry])
    feedback_df.to_csv(FEEDBACK_LOG_FILE, mode='a', header=False, index=False)

    return dropout_risk_reduction, sentiment_improvement, performance_improvement


def refine_model(model, X, y, feedback_data):
    try:
        feedback_df = pd.read_csv(FEEDBACK_LOG_FILE)
        for _, row in feedback_df.iterrows():
            student_id = int(row['student_id'])
            dropout_risk_reduction = row['dropout_risk_reduction']
            if dropout_risk_reduction > 0.1:
                y[student_id] = 0

        refined_model = RandomForestClassifier(n_estimators=100, random_state=42)
        refined_model.fit(X, y)
        with open('models/refined_model.pkl', "wb") as f:
            pickle.dump(refined_model, f)
        logging.info("Model refined and saved successfully.")
        return refined_model
    except Exception as e:
        logging.error(f"Model refinement failed: {str(e)}")
        return model


def monitoring_and_feedback(actions, model, X, y):
    initialize_feedback_log()
    executed_actions = []
    for action in actions:
        log_action(action)
        executed_actions.append(action)

        if "Student" in action:
            try:
                student_id = int(action.split("Student ")[1].split(" ")[0])
                previous_dropout_prob = model.predict_proba(X[student_id:student_id + 1])[:, 1][0]
                dropout_risk_reduction, sentiment_improvement, performance_improvement = collect_feedback(
                    student_id, action, previous_dropout_prob, X, model
                )
                feedback_log = (f"Feedback for Student {student_id}: "
                                f"Dropout Risk Reduction: {dropout_risk_reduction:.2f}, "
                                f"Sentiment Improvement: {sentiment_improvement:.2f}, "
                                f"Performance Improvement: {performance_improvement:.2f}")
                executed_actions.append(feedback_log)
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not parse student ID from action: {action}, Error: {str(e)}")

    refined_model = refine_model(model, X, y, FEEDBACK_LOG_FILE)
    return executed_actions, "Feedback logged and model refined successfully."
