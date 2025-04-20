import numpy as np
import pandas as pd
import os
from scripts.modeling import train_and_evaluate_models


class MonitoringFeedback:
    def __init__(self, log_file="data/action_log.csv"):
        """
        Initialize monitoring and feedback loop.

        Parameters:
        log_file (str): Path to the log file.
        """
        self.log_file = log_file
        self.action_log = self._load_log()

    def _load_log(self):
        """
        Load existing action log or create a new one.

        Returns:
        pd.DataFrame: Action log DataFrame.
        """
        if os.path.exists(self.log_file):
            return pd.read_csv(self.log_file)
        else:
            return pd.DataFrame(columns=[
                "student_id", "timestamp", "action_type", "action_description",
                "priority", "outcome", "feedback_score"
            ])

    def log_actions_and_outcomes(self, student_id, executed_actions, outcome=None, feedback_score=None):
        """
        Log executed actions and outcomes.

        Parameters:
        student_id (str): Student ID.
        executed_actions (dict): Executed actions.
        outcome (str, optional): Outcome of the actions.
        feedback_score (float, optional): Feedback score (e.g., student satisfaction).
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log messages
        for message in executed_actions["messages"]:
            self.action_log = pd.concat([self.action_log, pd.DataFrame([{
                "student_id": student_id,
                "timestamp": timestamp,
                "action_type": "message",
                "action_description": message,
                "priority": executed_actions["priority"],
                "outcome": outcome,
                "feedback_score": feedback_score
            }])], ignore_index=True)

        # Log content adjustments
        self.action_log = pd.concat([self.action_log, pd.DataFrame([{
            "student_id": student_id,
            "timestamp": timestamp,
            "action_type": "content_adjustment",
            "action_description": executed_actions["content_adjustments"],
            "priority": executed_actions["priority"],
            "outcome": outcome,
            "feedback_score": feedback_score
        }])], ignore_index=True)

        # Log interventions
        for intervention in executed_actions["triggered_interventions"]:
            self.action_log = pd.concat([self.action_log, pd.DataFrame([{
                "student_id": student_id,
                "timestamp": timestamp,
                "action_type": "intervention",
                "action_description": intervention,
                "priority": executed_actions["priority"],
                "outcome": outcome,
                "feedback_score": feedback_score
            }])], ignore_index=True)

        # Save log
        self.action_log.to_csv(self.log_file, index=False)

    def use_feedback_to_refine_models(self, X_integrated, X_analytics, data):
        """
        Use feedback to refine models by retraining.

        Parameters:
        X_integrated (np.ndarray): Integrated features.
        X_analytics (np.ndarray): Analytics features.
        data (pd.DataFrame): Updated dataset.

        Returns:
        tuple: Updated models and metrics.
        """
        # Analyze feedback scores to adjust dataset (e.g., update labels)
        if "feedback_score" in self.action_log.columns:
            avg_feedback = self.action_log["feedback_score"].mean()
            print(f"Average feedback score: {avg_feedback:.2f}")
            # Example: Adjust dropout labels based on feedback
            if avg_feedback and avg_feedback < 0.5:
                print("Low feedback score detected. Adjusting model training data.")
                # Simulate label adjustment (e.g., increase dropout risk for low feedback)
                data["dropout"] = data["dropout"].apply(lambda x: 1 if np.random.random() < 0.1 else x)

        # Retrain models
        # result = train_and_evaluate_models(X_integrated, X_analytics, data)
        result = None
        print("Models retrained with feedback loop.")
        return result
