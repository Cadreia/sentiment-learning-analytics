import pickle
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.data_preprocessing import get_spacy_embedding


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model


# Load models
dropout_model_analytics = load_model("models/dropout_model_analytics.pkl")
dropout_model_integrated = load_model("models/dropout_model_integrated.pkl")
engagement_model = load_model("models/engagement_model.pkl")
performance_model = load_model("models/performance_model.pkl")

# Define feature sets
numerical_cols = ["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
                  "Quizzes_Avg", "Participation_Score", "Projects_Score", 'Total_Score',
                  "Study_Hours_per_Week", "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Age"]
categorical_cols = ["Gender", "Department", "Grade", "Extracurricular_Activities",
                    "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level"]

# Load PCA models
pca_analytics = joblib.load("models/pca/pca_analytics.joblib")
pca_integrated = joblib.load("models/pca/pca_integrated.joblib")


def predict(student_data, coursecontent_text="", labwork_text="", coursecontent_sentiment=None, labwork_sentiment=None, model_type="integrated", predict_type="dropout"):
    # Ensure all required columns are present
    for col in numerical_cols + categorical_cols:
        if col not in student_data.columns:
            student_data[col] = 0  # Default encoded mode

    if model_type == "analytics":
        # Use numerical + categorical features with pca_analytics
        features = numerical_cols + categorical_cols
        X = student_data[features].to_numpy(dtype=np.float32)
        X_reduced = pca_analytics.transform(X)
        return dropout_model_analytics.predict(X_reduced)[0]

    elif model_type == "integrated":
        # Generate embeddings
        coursecontent_embedding = get_spacy_embedding(coursecontent_text)
        labwork_embedding = get_spacy_embedding(labwork_text)

        # Create embedding DataFrames
        embedding_dim = 300
        coursecontent_features = [f"coursecontent_embedding_{i}" for i in range(embedding_dim)]
        labwork_features = [f"labwork_embedding_{i}" for i in range(embedding_dim)]
        coursecontent_df = pd.DataFrame([coursecontent_embedding], columns=coursecontent_features).astype(np.float32)
        labwork_df = pd.DataFrame([labwork_embedding], columns=labwork_features).astype(np.float32)

        # Combine all features
        student_data["coursecontent_sentiment_score"] = coursecontent_sentiment
        student_data["labwork_sentiment_score"] = labwork_sentiment

        X_integrated = pd.concat([
            student_data[numerical_cols + categorical_cols +
                         ["coursecontent_sentiment_score", "labwork_sentiment_score"]],
            coursecontent_df,
            labwork_df
        ], axis=1).astype(np.float32)

        # Handle any NaNs
        X_integrated.fillna(0, inplace=True)

        # Apply PCA
        X_reduced = pca_integrated.transform(X_integrated.to_numpy())

        match predict_type:
            case "dropout":
                return dropout_model_integrated.predict(X_reduced)[0]
            case "engagement":
                # Predict encoded engagement
                le = LabelEncoder()
                le.fit(["Low", "High"])  # Match modeling.py's encoding
                encoded_pred = engagement_model.predict(X_reduced)[0]
                return le.inverse_transform([encoded_pred])[0]
            case "performance":
                return performance_model.predict(X_reduced)[0]
            case _:
                raise ValueError("predict_type must be 'dropout' or 'engagement' or 'performance'")

    else:
        raise ValueError("model_type must be 'analytics' or 'integrated'")


def recommend_action(coursecontent_polarity, labwork_polarity, total_score):
    if (coursecontent_polarity < 0 or labwork_polarity < 0) and total_score < 0.5:
        return "Recommend additional resources"
    elif (coursecontent_polarity > 0 and labwork_polarity > 0) and total_score > 0.8:
        return "Suggest advanced content"
    return "No action"
