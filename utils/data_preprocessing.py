# utils/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

from scripts.sentiment_analysis import analyze_sentiment


@st.cache_data
def load_data(analytics_file, feedback_file):
    analytics_df = pd.read_csv(analytics_file)
    feedback_df = pd.read_csv(feedback_file)
    return analytics_df, feedback_df


def preprocess_data(analytics_df, feedback_df, tokenizer, model):
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
                        'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']
    for col in categorical_cols:
        analytics_df[col] = le.fit_transform(analytics_df[col].astype(str))

    scaler = StandardScaler()
    numerical_cols = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
                      'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
                      'Study_Hours_per_Week', 'Age', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
    analytics_df[numerical_cols] = scaler.fit_transform(analytics_df[numerical_cols])

    feedback_df['coursecontent_sentiment'] = feedback_df['coursecontent'].apply(
        lambda x: analyze_sentiment(str(x), tokenizer, model)[0] if pd.notnull(x) else 1)
    feedback_df['labwork_sentiment'] = feedback_df['labwork'].apply(
        lambda x: analyze_sentiment(str(x), tokenizer, model)[0] if pd.notnull(x) else 1)

    n_assessments = len(analytics_df)
    n_feedback = len(feedback_df)
    feedback_indices = np.arange(n_feedback)
    feedback_indices = np.tile(feedback_indices, (n_assessments // n_feedback) + 1)[:n_assessments]

    analytics_df["feedback_index"] = feedback_indices
    merged_df = analytics_df.merge(
        feedback_df[['coursecontent_sentiment', 'labwork_sentiment']].reset_index().rename(
            columns={"index": "feedback_index"}),
        on="feedback_index",
        how="left"
    )
    merged_df.fillna({'coursecontent_sentiment': 1, 'labwork_sentiment': 1}, inplace=True)
    merged_df.drop(columns=["feedback_index"], inplace=True)

    features = numerical_cols + categorical_cols + ['coursecontent_sentiment', 'labwork_sentiment']
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(merged_df[features])

    return merged_df, reduced_features, features
