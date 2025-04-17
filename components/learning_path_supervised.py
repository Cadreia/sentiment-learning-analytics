# components/learning_path_supervised.py
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd


def learning_path_supervised(X_integrated, groups, merged_df):
    """
    Perform supervised learning path analysis using precomputed predictions from merged_df.

    Parameters:
    - X_integrated (np.ndarray): Reduced integrated features (with sentiment).
    - dropout_model: Trained model for dropout prediction (not used since predictions are precomputed).
    - performance_model: Trained model for performance prediction (not used since predictions are precomputed).
    - engagement_model: Trained model for engagement prediction (not used since predictions are precomputed).
    - groups (np.ndarray): Precomputed cluster groups for students.
    - merged_df (pd.DataFrame): DataFrame containing precomputed predictions and original data.

    Returns:
    - performance_pred (np.ndarray): Predicted performance scores (from merged_df).
    - engagement_pred (np.ndarray): Predicted engagement labels (from merged_df).
    - groups (np.ndarray): Cluster groups (passed through unchanged).
    - outliers (np.ndarray): Binary array indicating outliers (1 for outliers, 0 otherwise).
    """
    # Extract precomputed predictions from merged_df
    try:
        dropout_pred_int = merged_df["dropout_pred_int"].values
        dropout_pred_ana = merged_df["dropout_pred_ana"].values
        performance_pred = merged_df["performance_pred"].values
        engagement_pred = merged_df["engagement_pred"].values
    except KeyError as e:
        raise KeyError(f"Missing expected prediction column in merged_df: {e}")

    # Debug: Print shapes and types of predictions
    print("Shape of dropout_pred_int:", dropout_pred_int.shape)
    print("Shape of dropout_pred_ana:", dropout_pred_ana.shape)
    print("Shape of performance_pred:", performance_pred.shape)
    print("Shape of engagement_pred:", engagement_pred.shape)

    # Detect outliers using Isolation Forest on X_integrated
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X_integrated)
    outliers = (outliers == -1).astype(int)  # Convert to binary: 1 for outliers, 0 for inliers

    # Debug: Print outlier detection results
    print("Number of outliers detected:", np.sum(outliers))

    # Groups are passed through unchanged (already computed in modeling.py)
    # Return the predictions, groups, and outliers
    return performance_pred, engagement_pred, groups, outliers
