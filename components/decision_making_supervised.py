# components/decision_making_supervised.py
def decision_making_supervised(student_data, model, performance_pred, engagement_pred, groups, outliers, threshold=0.5):
    dropout_probs = model.predict_proba(student_data)[:, 1]
    actions = []
    for i, (prob, performance, engagement, group, outlier) in enumerate(
            zip(dropout_probs, performance_pred, engagement_pred, groups, outliers)):
        student_id = i
        feedback = f"Provide Feedback to Student {student_id}: "
        if prob > threshold:
            feedback += "Your recent performance suggests focusing on time management."
        else:
            feedback += "Great job! Keep up the good work."
        actions.append(feedback)

        if prob > threshold or outlier:
            intervention = f"Suggest Intervention for Student {student_id}: Extra tutoring sessions (Dropout Risk: {prob:.2f})"
            actions.append(intervention)

    return actions
