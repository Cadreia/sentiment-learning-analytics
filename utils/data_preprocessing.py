import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from scripts.sentiment_analysis import get_sentiment_scores, load_sentiment_pipeline
from scripts.shap import get_shap_results
import os
import joblib

# Ensure NLTK data is downloaded
import nltk

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


# Load spaCy model with caching
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.error("SpaCy model 'en_core_web_md' not found. Run python -m spacy download en_core_web_md to install it.")
        st.warning("Using zero vectors for text embeddings as a fallback.")
        return None


nlp = load_spacy_model()

# Define numerical and categorical columns at module level for export
numerical_cols = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
                  'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
                  'Study_Hours_per_Week', 'Age', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
categorical_cols = ['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
                    'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']


# Get spaCy embedding
def get_spacy_embedding(text):
    """
    Compute the average word embedding for a text string using SpaCy's en_core_web_md model.

    Parameters:
    - text (str): The input text to embed.

    Returns:
    - np.ndarray: A 300-dimensional vector (average of word embeddings, excluding stop words).
    """
    if not nlp or pd.isna(text) or str(text).strip() == "":
        return np.zeros(300)
    doc = nlp(str(text))
    # Compute the average vector of non-stop words
    vectors = [token.vector for token in doc if not token.is_stop]
    if not vectors:  # If all tokens are stop words, return zero vector
        return np.zeros(300)
    return np.mean(vectors, axis=0)


@st.cache_data
def load_data(analytics_file, feedback_file):
    analytics_df = pd.read_csv(analytics_file, header=0)
    feedback_df = pd.read_csv(feedback_file, header=0)
    return analytics_df, feedback_df


def normalize_data(df, numerical_cols, categorical_cols):
    """
    Normalize numerical and categorical columns in a DataFrame by imputing missing values,
    encoding categorical variables, and scaling numerical features to [0, 1].

    Parameters:
    - df (pd.DataFrame): Input DataFrame to normalize.
    - numerical_cols (list): List of numerical column names.
    - categorical_cols (list): List of categorical column names.

    Returns:
    - pd.DataFrame: Normalized DataFrame with imputed, encoded, and scaled values.
    """
    # Create a copy to avoid modifying the original
    df_normalized = df.copy()

    # Validate columns
    missing_num_cols = [col for col in numerical_cols if col not in df_normalized.columns]
    missing_cat_cols = [col for col in categorical_cols if col not in df_normalized.columns]
    if missing_num_cols or missing_cat_cols:
        missing = missing_num_cols + missing_cat_cols
        st.warning(f"Missing columns in DataFrame: {', '.join(missing)}. Skipping normalization for these columns.")
        numerical_cols = [col for col in numerical_cols if col in df_normalized.columns]
        categorical_cols = [col for col in categorical_cols if col in df_normalized.columns]

    # Impute missing numerical values with mean
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df_normalized[numerical_cols] = num_imputer.fit_transform(df_normalized[numerical_cols])

    # Impute missing categorical values with most frequent
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_normalized[categorical_cols] = cat_imputer.fit_transform(df_normalized[categorical_cols])

        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_cols:
            df_normalized[col] = le.fit_transform(df_normalized[col].astype(str))

    # Scale numerical features to [0, 1]
    if numerical_cols:
        scaler = MinMaxScaler()
        df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])

    return df_normalized


def compute_sentiment_scores(texts, sentiment_pipeline=None):
    """
    Compute sentiment scores for a list of texts using a preloaded sentiment pipeline.

    Parameters:
    - texts (list): List of text strings.
    - sentiment_pipeline: Preloaded sentiment analysis pipeline (optional).

    Returns:
    - list: List of sentiment scores.
    """
    if sentiment_pipeline is None:
        sentiment_pipeline = load_sentiment_pipeline()
    return get_sentiment_scores([str(text) if not pd.isna(text) else "" for text in texts], sentiment_pipeline)


def prepare_student_features(student_data, coursecontent_text, labwork_text, sentiment_pipeline=None):
    """
    Prepare student features for analysis, including normalization, sentiment scoring, and embeddings.

    Parameters:
    - student_data (pd.DataFrame): Student data.
    - coursecontent_text (str): Course content feedback.
    - labwork_text (str): Lab work feedback.
    - numerical_cols (list): Numerical columns.
    - categorical_cols (list): Categorical columns.
    - sentiment_pipeline: Preloaded sentiment analysis pipeline (optional).

    Returns:
    - tuple: (normalized_data, X_integrated)
        - normalized_data (pd.DataFrame): Normalized student data with sentiment scores.
        - X_integrated (np.ndarray): Integrated features array for unsupervised decision-making.
    """
    # Normalize the core student data
    normalized_data = normalize_data(student_data, numerical_cols, categorical_cols)

    if coursecontent_text == "" or labwork_text == "" or sentiment_pipeline is None:
        X_integrated = None
    else:
        # Compute sentiment scores
        coursecontent_polarity = compute_sentiment_scores([coursecontent_text], sentiment_pipeline)[0]
        labwork_polarity = compute_sentiment_scores([labwork_text], sentiment_pipeline)[0]

        # Add sentiment scores to normalized data
        normalized_data["coursecontent_sentiment_score"] = coursecontent_polarity
        normalized_data["labwork_sentiment_score"] = labwork_polarity

        # Compute embeddings
        embedding_dim = 300
        coursecontent_embedding = get_spacy_embedding(coursecontent_text)
        labwork_embedding = get_spacy_embedding(labwork_text)
        coursecontent_features = [f"coursecontent_embedding_{i}" for i in range(embedding_dim)]
        labwork_features = [f"labwork_embedding_{i}" for i in range(embedding_dim)]
        coursecontent_df = pd.DataFrame([coursecontent_embedding], columns=coursecontent_features,
                                        index=student_data.index).astype(np.float32)
        labwork_df = pd.DataFrame([labwork_embedding], columns=labwork_features, index=student_data.index).astype(
            np.float32)

        # Combine all features for X_integrated
        X_integrated = pd.concat([
            normalized_data[numerical_cols + categorical_cols +
                            ["coursecontent_sentiment_score", "labwork_sentiment_score"]],
            coursecontent_df,
            labwork_df
        ], axis=1, ignore_index=False).astype(np.float32)

        # Handle any NaNs
        X_integrated.fillna(0, inplace=True)

        # Convert to Numpy array
        X_integrated.to_numpy()

    return normalized_data, X_integrated


@st.cache_data
def preprocess_data(analytics_df, feedback_df):
    # Preprocess analytics-only data
    analytics_only_df = analytics_df.copy()
    analytics_only_df = normalize_data(analytics_only_df, numerical_cols, categorical_cols)
    scaler = MinMaxScaler()
    analytics_only_df[numerical_cols] = scaler.fit_transform(analytics_only_df[numerical_cols])
    analytics_features = numerical_cols + categorical_cols

    X_analytics = analytics_only_df[analytics_features].to_numpy(dtype=np.float32)
    if not np.isfinite(X_analytics).all():
        raise ValueError("NaNs or infinite values in analytics-only data before PCA.")

    pca_analytics = PCA(n_components=0.95)
    reduced_analytics = pca_analytics.fit_transform(X_analytics)

    # Save PCA model for prediction
    pca_path = "models/pca/pca_analytics.joblib"
    pca_dir = os.path.dirname(pca_path)
    os.makedirs(pca_dir, exist_ok=True)         # Create the directory if it doesn't exist
    joblib.dump(pca_analytics, pca_path)
    print(f"Analytics PCA model saved to {pca_path}")

    # Preprocess full analytics data
    analytics_df = normalize_data(analytics_df, numerical_cols, categorical_cols)

    # Preprocess feedback data
    feedback_df = feedback_df[["coursecontent.1", "labwork.1"]]
    feedback_df.rename(columns={"coursecontent.1": "coursecontent_text", "labwork.1": "labwork_text"}, inplace=True)

    n_assessments = len(analytics_df)
    n_feedback = len(feedback_df)
    feedback_indices = np.arange(n_feedback)
    feedback_indices = np.tile(feedback_indices, (n_assessments // n_feedback) + 1)[:n_assessments]

    analytics_df["feedback_index"] = feedback_indices
    merged_df = analytics_df.merge(
        feedback_df.reset_index().rename(columns={"index": "feedback_index"}),
        on="feedback_index",
        how="left"
    )
    merged_df.drop(columns=["feedback_index"], inplace=True)
    merged_df[["coursecontent_text", "labwork_text"]] = merged_df[["coursecontent_text", "labwork_text"]].fillna("")

    merged_df.to_csv("data/integrated_data.csv", index=False, header=True)
    print("Integrated dataset saved to data/integrated_data.csv")

    # Text preprocessing
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError as e:
        print(f"Error loading stopwords: {e}")
        stop_words = set()

    def preprocess_text(text):
        try:
            if pd.isna(text) or text == "":
                return ""
            tokens = word_tokenize(str(text).lower())
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
            return " ".join(tokens)
        except LookupError as e:
            print(f"Error in tokenization: {e}")
            return str(text).lower()

    merged_df["coursecontent_text_cleaned"] = merged_df["coursecontent_text"].apply(preprocess_text)
    merged_df["labwork_text_cleaned"] = merged_df["labwork_text"].apply(preprocess_text)

    # Generate sentiment scores
    sentiment_pipeline = load_sentiment_pipeline()
    if 'coursecontent_sentiment_score' not in merged_df.columns or merged_df[
        'coursecontent_sentiment_score'].isnull().all():
        merged_df['coursecontent_sentiment_score'] = compute_sentiment_scores(
            merged_df["coursecontent_text"].fillna("").tolist(),
            sentiment_pipeline
        )

    if 'labwork_sentiment_score' not in merged_df.columns or merged_df['labwork_sentiment_score'].isnull().all():
        merged_df['labwork_sentiment_score'] = compute_sentiment_scores(
            merged_df["labwork_text"].fillna("").tolist(),
            sentiment_pipeline
        )

    # Dropout label
    def determine_dropout(row):
        if row["Grade"] in [2, 3, 4] and row["Total_Score"] < 0.6 and row["Attendance (%)"] < 0.6 and row[
            "Stress_Level (1-10)"] > 0.4:
            return 1
        return 0

    merged_df["dropout"] = merged_df.apply(determine_dropout, axis=1)
    print(merged_df["dropout"])
    merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])

    # Cache embeddings
    embeddings_path = "data/embeddings.npz"
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
        coursecontent_embeddings = embeddings["coursecontent"]
        labwork_embeddings = embeddings["labwork"]
    else:
        coursecontent_embeddings = np.array([
            get_spacy_embedding(text) for text in merged_df["coursecontent_text"]
        ], dtype=np.float32)
        labwork_embeddings = np.array([
            get_spacy_embedding(text) for text in merged_df["labwork_text"]
        ], dtype=np.float32)
        np.savez(embeddings_path, coursecontent=coursecontent_embeddings, labwork=labwork_embeddings)

    embedding_dim = 300
    coursecontent_features = [f"coursecontent_embedding_{i}" for i in range(embedding_dim)]
    labwork_features = [f"labwork_embedding_{i}" for i in range(embedding_dim)]

    coursecontent_embeddings_df = pd.DataFrame(coursecontent_embeddings, columns=coursecontent_features).astype(
        np.float32)
    labwork_embeddings_df = pd.DataFrame(labwork_embeddings, columns=labwork_features).astype(np.float32)

    # Combine features
    integrated_features = (
            numerical_cols + categorical_cols +
            ["coursecontent_sentiment_score", "labwork_sentiment_score"] +
            coursecontent_features + labwork_features
    )

    X_integrated = pd.concat([
        merged_df[numerical_cols + categorical_cols +
                  ["coursecontent_sentiment_score", "labwork_sentiment_score"]].reset_index(drop=True),
        coursecontent_embeddings_df.reset_index(drop=True),
        labwork_embeddings_df.reset_index(drop=True)
    ], axis=1)

    # Ensure all columns are numeric and convert to float32
    X_integrated = X_integrated.apply(pd.to_numeric, errors='coerce').astype(np.float32)

    # Replace any remaining NaNs with 0
    X_integrated.fillna(0, inplace=True)

    X_integrated_array = X_integrated.to_numpy()
    if not np.isfinite(X_integrated_array).all():
        raise ValueError("NaNs or infinite values in integrated data before PCA.")

    pca_integrated = PCA(n_components=0.95)
    reduced_integrated = pca_integrated.fit_transform(X_integrated_array)

    # Save PCA model for integrated features
    pca_integrated_path = "models/pca/pca_integrated.joblib"
    joblib.dump(pca_integrated, pca_integrated_path)
    print(f"Integrated PCA model saved to {pca_integrated_path}")

    integrated_data = merged_df.copy()
    integrated_data.to_csv("data/integrated_data_polarity.csv", index=False, header=True)
    print("Integrated dataset with sentiment polarity saved to data/integrated_data_polarity.csv")

    return merged_df, reduced_integrated, reduced_analytics, integrated_features, analytics_features, X_integrated
