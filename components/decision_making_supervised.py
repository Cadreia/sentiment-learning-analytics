# def supervised_decision_making(student_data, model, performance_pred, engagement_pred, groups, outliers, threshold=0.5):
#     dropout_probs = model.predict_proba(student_data)[:, 1]
#     actions = []
#     for i, (prob, performance, engagement, group, outlier) in enumerate(
#             zip(dropout_probs, performance_pred, engagement_pred, groups, outliers)):
#         student_id = i
#         feedback = f"Provide Feedback to Student {student_id}: "
#         if prob > threshold:
#             feedback += "Your recent performance suggests focusing on time management."
#         else:
#             feedback += "Great job! Keep up the good work."
#         actions.append(feedback)
#
#         if prob > threshold or outlier:
#             intervention = f"Suggest Intervention for Student {student_id}: Extra tutoring sessions (Dropout Risk: {prob:.2f})"
#             actions.append(intervention)
#
#     return actions


from scripts.predictions import predict
from scripts.sentiment_analysis import get_sentiment_scores, load_sentiment_pipeline


def supervised_decision_making(student_data, coursecontent_text, labwork_text, student_id, model_type="integrated"):
    """
    Perform supervised decision-making using model predictions.

    Parameters:
    student_data (pd.DataFrame): DataFrame with student features.
    coursecontent_text (str): Course content feedback.
    labwork_text (str): Lab work feedback.
    student_id (str): Student ID.
    model_type (str): Model type for dropout prediction ("analytics" or "integrated").

    Returns:
    dict: Decisions including predictions, feedback, and interventions.
    """
    # # Compute sentiment scores if not already in student_data
    # sentiment_pipeline = load_sentiment_pipeline()
    # coursecontent_sentiment = get_sentiment_scores([coursecontent_text], sentiment_pipeline)[0]
    # labwork_sentiment = get_sentiment_scores([labwork_text], sentiment_pipeline)[0]

    # # Get sentiment scores if already in student_data
    # coursecontent_sentiment = student_data['coursecontent_sentiment_score']
    # labwork_sentiment = student_data['labwork_sentiment_score']

    # Make predictions using the unified predict function
    dropout_risk = predict(
        student_data,
        coursecontent_text=coursecontent_text,
        labwork_text=labwork_text,
        # coursecontent_sentiment=coursecontent_sentiment,
        # labwork_sentiment=labwork_sentiment,
        model_type=model_type,
        predict_type="dropout"
    )
    performance_pred = predict(
        student_data,
        coursecontent_text=coursecontent_text,
        labwork_text=labwork_text,
        model_type=model_type,
        predict_type="performance"
    )
    engagement_pred = predict(
        student_data,
        coursecontent_text=coursecontent_text,
        labwork_text=labwork_text,
        model_type=model_type,
        predict_type="engagement"
    )

    # Provide feedback based on predictions
    feedback = provide_feedback(dropout_risk, performance_pred, engagement_pred, student_id)

    # Suggest interventions
    interventions = suggest_interventions(dropout_risk, performance_pred, engagement_pred)

    return {
        "student_id": student_id,
        "predictions": {
            "dropout_risk": dropout_risk,
            "performance_pred": performance_pred,
            "engagement_pred": engagement_pred
        },
        "dropout_risk": dropout_risk,  # Kept for compatibility
        "performance_pred": performance_pred,  # Kept for compatibility
        "engagement_pred": engagement_pred,  # Kept for compatibility
        "feedback": feedback,
        "interventions": interventions
    }


def provide_feedback(dropout_risk, performance_pred, engagement_pred, student_id):
    """
    Provide feedback based on predictions.

    Parameters:
    dropout_risk (int): Predicted dropout risk (0 or 1).
    performance_pred (float): Predicted performance (Total_Score).
    engagement_pred (str): Predicted engagement ("High" or "Low").
    student_id (str): Student ID.

    Returns:
    str: Feedback message.
    """
    feedback = f"Feedback for Student {student_id}:\n"
    if dropout_risk == 1:
        feedback += "\n - High dropout risk detected. Immediate support recommended.\n"
    else:
        feedback += "\n - Low dropout risk. Continue monitoring.\n"

    feedback += f"\n - Performance Prediction: {performance_pred:.2f}. "
    if performance_pred < 0.6:
        feedback += "Consider additional academic support.\n"
    else:
        feedback += "Performance is satisfactory.\n"

    feedback += f"\n - Engagement Level: {engagement_pred}. "
    if engagement_pred == "Low":
        feedback += "Engagement needs improvement.\n"
    else:
        feedback += "Engagement is strong.\n"

    return feedback


def suggest_interventions(dropout_risk, performance_pred, engagement_pred):
    """
    Suggest interventions based on predictions.

    Parameters:
    dropout_risk (int): Predicted dropout risk (0 or 1).
    performance_pred (float): Predicted performance (Total_Score).
    engagement_pred (str): Predicted engagement ("High" or "Low").

    Returns:
    list: List of suggested interventions.
    """
    interventions = []
    if dropout_risk == 1:
        interventions.append("Schedule one-on-one counseling session.")
    if performance_pred < 0.6:
        interventions.append("Assign peer tutor for additional support.")
    if engagement_pred == "Low":
        interventions.append("Incorporate interactive activities to boost engagement.")
    if not interventions:
        interventions.append("No immediate interventions needed.")
    return interventions
