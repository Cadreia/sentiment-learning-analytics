# scripts/actions.py
import pandas as pd
from utils.preprocess import preprocess_data
from scripts.modeling import train_and_evaluate_models
from components.decision_making_unsupervised import decision_making_unsupervised
from components.decision_making_supervised import decision_making_supervised
from components.fusion_adaptation import fusion_adaptation
from components.decision_fusion import decision_fusion
from components.action_execution import action_execution
from components.monitoring import monitoring_and_feedback


def bert_sentiment_to_polarity(sentiment_score):
    if sentiment_score == 0:  # Negative
        return -1
    elif sentiment_score == 1:  # Neutral
        return 0
    else:  # Positive
        return 1


def run_actions(analytics_data, feedback_data):
    integrated_data, reduced_features, features = preprocess_data(analytics_data, feedback_data)

    if "coursecontent_polarity" not in integrated_data.columns or "labwork_polarity" not in integrated_data.columns:
        integrated_data["coursecontent_polarity"] = integrated_data["coursecontent_sentiment"].apply(
            bert_sentiment_to_polarity)
        integrated_data["labwork_polarity"] = integrated_data["labwork_sentiment"].apply(bert_sentiment_to_polarity)

    # Train models and get predictions
    (dropout_model, performance_model, engagement_model, groups, _, _, _, _, _, _) = train_and_evaluate_models(
        reduced_features, integrated_data)

    integrated_data["dropout_pred"] = integrated_data["dropout_pred"]
    integrated_data["performance_pred"] = integrated_data["performance_pred"]
    integrated_data["engagement_pred"] = integrated_data["engagement_pred"]
    integrated_data["learning_path"] = groups
    integrated_data["learning_path"] = integrated_data["learning_path"].apply(
        lambda x: "Needs Support" if x == 2 else ("Advanced" if x == 0 else "Standard")
    )

    unsupervised_actions = decision_making_unsupervised(reduced_features, groups, integrated_data["anomaly"])
    supervised_actions = decision_making_supervised(reduced_features, dropout_model,
                                                    integrated_data["performance_pred"],
                                                    integrated_data["engagement_pred"],
                                                    groups,
                                                    integrated_data["outlier"])
    fused_actions = fusion_adaptation(unsupervised_actions, supervised_actions)
    final_actions = decision_fusion(fused_actions)
    executed_actions = action_execution(final_actions)
    y = integrated_data["dropout"]
    executed_actions_with_feedback, feedback_result = monitoring_and_feedback(executed_actions, dropout_model,
                                                                              reduced_features, y)

    monitoring_log = pd.DataFrame({
        "Student_ID": integrated_data["Student_ID"],
        "Action": [action for action in executed_actions_with_feedback if
                   "Student" in action and "Feedback for Student" not in action],
        "Feedback": [action for action in executed_actions_with_feedback if "Feedback for Student" in action]
    })
    monitoring_log.to_csv("data/monitoring_log.csv", index=False)

    return integrated_data, executed_actions_with_feedback, feedback_result


if __name__ == "__main__":
    analytics_data = pd.read_csv("data/student_assessments.csv")
    feedback_data = pd.read_csv("data/student_feedback.csv")
    run_actions(analytics_data, feedback_data)
