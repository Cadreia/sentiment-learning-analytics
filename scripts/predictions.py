# predictions.py (Updated Version)
import pickle
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scripts.sentiment_analysis import load_sentiment_pipeline
from utils.data_preprocessing import prepare_student_features, numerical_cols, categorical_cols

# Define model paths
MODEL_PATHS = {
    "dropout_analytics": "models/dropout_model_analytics.pkl",
    "dropout_integrated": "models/dropout_model_integrated.pkl",
    "engagement": "models/engagement_model.pkl",
    "performance": "models/performance_model.pkl"
}

# Define pca model paths
PCA_MODEL_PATHS = {
    "pca_analytics": "models/pca/pca_analytics.joblib",
    "pca_integrated": "models/pca/pca_integrated.joblib",
}

# Dictionary to store loaded models (will be populated on demand)
MODELS = {}
PCA_MODELS = {}

# Initialize LabelEncoder for engagement predictions
ENGAGEMENT_ENCODER = LabelEncoder()
ENGAGEMENT_ENCODER.fit(["Low", "High"])  # Match modeling.py's encoding

# Load sentiment pipeline once at module level
sentiment_pipeline = load_sentiment_pipeline()


def load_model(model_name):
    """
    Load a model on demand if not already loaded.

    Parameters:
    - model_name (str): Name of the model to load.

    Returns:
    - Model object if loaded, None if the model file is missing.
    """
    if model_name in MODELS:
        return MODELS[model_name]

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown model name: {model_name}")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            MODELS[model_name] = model
            return model
    else:
        return None


def load_pca_model(model_name):
    """
    Load a model on demand if not already loaded.

    Parameters:
    - model_name (str): Name of the model to load.

    Returns:
    - Model object if loaded, None if the model file is missing.
    """
    if model_name in PCA_MODELS:
        return PCA_MODELS[model_name]

    pca_model_path = PCA_MODEL_PATHS.get(model_name)
    if not pca_model_path:
        raise ValueError(f"Unknown PCA model name: {model_name}")

    if os.path.exists(pca_model_path):
        return joblib.load(pca_model_path)
    else:
        return None


def apply_pca(X, model_type):
    """
    Apply PCA transformation based on the model type.

    Parameters:
    - X (np.ndarray): Feature array.
    - model_type (str): Type of model ('analytics' or 'integrated').

    Returns:
    - np.ndarray: PCA-transformed features.
    """
    if model_type == "analytics":
        pca_analytics = load_pca_model("pca_analytics")
        return pca_analytics.transform(X.to_numpy())
    elif model_type == "integrated":
        pca_integrated = load_pca_model("pca_integrated")
        return pca_integrated.transform(X.to_numpy())
    else:
        raise ValueError("model_type must be 'analytics' or 'integrated'")


def predict(student_data, coursecontent_text="", labwork_text="", model_type="integrated", predict_type="dropout"):
    # Ensure all required columns are present
    for col in numerical_cols + categorical_cols:
        if col not in student_data.columns:
            student_data[col] = 0  # Default encoded mode

    # Determine the model name based on model_type and predict_type
    if model_type == "analytics":
        model_name = "dropout_analytics"
    elif model_type == "integrated":
        model_name = "dropout_integrated" if predict_type == "dropout" else predict_type
    else:
        raise ValueError("model_type must be 'analytics' or 'integrated'")

    # Load the model on demand
    model = load_model(model_name)
    if model is None:
        raise FileNotFoundError(f"Model '{model_name}' not found at {MODEL_PATHS[model_name]}. Please train the model "
                                f"first.")

    # Use prepare_student_features to get normalized and/or integrated data
    normalized_data, X_integrated = prepare_student_features(
        student_data,
        coursecontent_text,
        labwork_text,
        # coursecontent_sentiment,
        # labwork_sentiment,
        sentiment_pipeline
    )

    if model_type == "analytics":
        # Use numerical + categorical features with pca_analytics
        features = numerical_cols + categorical_cols
        X = student_data[features].to_numpy(dtype=np.float32)
        X_reduced = apply_pca(X, model_type)
        return model.predict(X_reduced)[0]

    elif model_type == "integrated":
        # Apply PCA
        X_reduced = apply_pca(X_integrated, model_type)

        match predict_type:
            case "dropout":
                return model.predict(X_reduced)[0], normalized_data, X_integrated
            case "engagement":
                # Predict encoded engagement
                encoded_pred = model.predict(X_reduced)[0]
                return ENGAGEMENT_ENCODER.inverse_transform([encoded_pred])[0]
            case "performance":
                return model.predict(X_reduced)[0]
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
