# components/learning_path_supervised.py
import numpy as np
from scripts.predictive_modeling import detect_outliers


def learning_path_supervised(X, model, data):
    performance_pred = data["Total_Score"].values
    engagement_pred = np.where(data["coursecontent_sentiment"] + data["labwork_sentiment"] > 2, "High", "Low")
    groups = data["learning_path"].values
    outliers = detect_outliers(X)
    return performance_pred, engagement_pred, groups, outliers
