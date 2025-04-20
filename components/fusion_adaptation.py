# def fusion_and_adaptation(unsupervised_actions, supervised_actions):
#     fused_actions = unsupervised_actions + supervised_actions
#     return fused_actions


def fusion_and_adaptation(unsupervised_decisions, supervised_decisions):
    """
    Merge unsupervised and supervised decisions and perform decision fusion.

    Parameters:
    unsupervised_decisions (dict): Decisions from unsupervised decision-making.
    supervised_decisions (dict): Decisions from supervised decision-making.

    Returns:
    dict: Fused decisions.
    """
    # Extract key unsupervised decisions
    student_id = unsupervised_decisions["student_id"]  # Extract student_id
    cluster = unsupervised_decisions["cluster"]
    is_anomaly = unsupervised_decisions["is_anomaly"]
    teaching_strategy = unsupervised_decisions["teaching_strategy"]
    review_flag = unsupervised_decisions["review_flag"]

    # Extract key unsupervised decisions
    dropout_risk = supervised_decisions["dropout_risk"]
    performance_pred = supervised_decisions["performance_pred"]
    engagement_pred = supervised_decisions["engagement_pred"]
    feedback = supervised_decisions["feedback"]
    interventions = supervised_decisions["interventions"]

    # Decision Fusion: Combine strategies and prioritize actions
    fused_strategy = decision_fusion(
        teaching_strategy, interventions, dropout_risk, performance_pred, engagement_pred
    )

    # Compile fused decisions
    fused_decisions = {
        "student_id": student_id,
        "cluster": cluster,
        "is_anomaly": is_anomaly,
        "review_flag": review_flag,
        "dropout_risk": dropout_risk,
        "performance_pred": performance_pred,
        "engagement_pred": engagement_pred,
        "feedback": feedback,
        "teaching_strategy": teaching_strategy,
        "interventions": interventions,
        "fused_strategy": fused_strategy
    }

    return fused_decisions


def decision_fusion(teaching_strategy, interventions, dropout_risk, performance_pred, engagement_pred):
    """
    Perform decision fusion to create a unified strategy.

    Parameters:
    teaching_strategy (str): Recommended teaching strategy from unsupervised decision.
    interventions (list): Suggested interventions from supervised decision.
    dropout_risk (int): Predicted dropout risk.
    performance_pred (float): Predicted performance.
    engagement_pred (str): Predicted engagement.

    Returns:
    dict: Fused strategy with prioritized actions.
    """
    fused_strategy = {
        "teaching_strategy": teaching_strategy,
        "interventions": interventions,
        "priority": "Low"
    }

    # Prioritize based on severity
    if dropout_risk == 1:
        fused_strategy["priority"] = "High"
        fused_strategy["interventions"].insert(0, "Immediate intervention required due to high dropout risk.")
    elif performance_pred < 0.6 or engagement_pred == "Low":
        fused_strategy["priority"] = "Medium"
        fused_strategy["interventions"].append("Monitor closely and adjust as needed.")

    return fused_strategy
