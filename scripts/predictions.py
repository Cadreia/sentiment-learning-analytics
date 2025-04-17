import pickle
import os


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model


# Load models
dropout_model = load_model("models/dropout_model_analytics.pkl")
engagement_model = load_model("models/engagement_model.pkl")
performance_model = load_model("models/performance_model.pkl")


def predict_performance(student_data):
    features = student_data[["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
                             "Quizzes_Avg", "Participation_Score", "Projects_Score",
                             "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Age"]]
    return performance_model.predict(features)[0]


def predict_dropout(student_data):
    features = student_data[["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
                             "Quizzes_Avg", "Participation_Score", "Projects_Score",
                             "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Age"]]
    return dropout_model.predict(features)[0]


def predict_engagement(student_data):
    features = student_data[["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
                             "Quizzes_Avg", "Participation_Score", "Projects_Score",
                             "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Age"]]
    return engagement_model.predict(features)[0]


def recommend_action(coursecontent_polarity, labwork_polarity, total_score):
    if (coursecontent_polarity < 0 or labwork_polarity < 0) and total_score < 0.5:
        return "Recommend additional resources"
    elif (coursecontent_polarity > 0 and labwork_polarity > 0) and total_score > 0.8:
        return "Suggest advanced content"
    return "No action"
