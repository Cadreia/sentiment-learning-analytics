# components/decision_making_unsupervised.py
# def unsupervised_decision_making(student_data, clusters, anomalies):
#     actions = []
#     for i, (cluster, anomaly) in enumerate(zip(clusters, anomalies)):
#         student_id = i
#         strategy = f"Adapt Teaching Strategy for Cluster {cluster}: "
#         if cluster == 0:
#             strategy += "Increase interactive content"
#         elif cluster == 1:
#             strategy += "Provide more practice problems"
#         else:
#             strategy += "Focus on conceptual understanding"
#         actions.append(strategy)
#
#         if anomaly:
#             action = f"Flag Student {student_id} for Review (Detected as Anomaly)"
#             actions.append(action)
#
#     return actions


from utils.clustering import cluster_students, detect_anomalies
import joblib
import numpy as np
import os


def unsupervised_decision_making(X_integrated, student_id, pre_trained_models=None, save_models=False):
    """
    Perform unsupervised decision-making using clustering and anomaly detection.

    Parameters:
    - X (np.ndarray): Feature array (shape: (n_samples, n_features)).
    - student_id (str): Identifier for the student or group being analyzed.
    - pre_trained_models (dict, optional): Pre-trained models for clustering and anomaly detection.
    - save_models (bool): Whether to save the trained models to disk.

    Returns:
    - dict: Decisions including cluster assignments, anomaly status, teaching strategy, and models (if trained).
    """
    # Determine if we're training new models or using pre-trained ones
    if pre_trained_models is None:
        # Train clustering model (KMeans)
        kmeans, cluster_labels = cluster_students(X_integrated)

        # Train anomaly detection model (IsolationForest)
        isolation_forest, anomaly_labels = detect_anomalies(X_integrated)
        is_anomaly = anomaly_labels == -1

        # Save models if requested
        if save_models:
            os.makedirs("models/unsupervised", exist_ok=True)
            joblib.dump(kmeans, f"models/unsupervised/kmeans_{student_id}.joblib")
            joblib.dump(isolation_forest, f"models/unsupervised/isolation_forest_{student_id}.joblib")

        models = {"kmeans": kmeans, "isolation_forest": isolation_forest}
    else:
        # Use pre-trained models
        kmeans = pre_trained_models["kmeans"]
        isolation_forest = pre_trained_models["isolation_forest"]
        cluster_labels = kmeans.predict(X_integrated)
        anomaly_labels = isolation_forest.predict(X_integrated)
        is_anomaly = anomaly_labels == -1
        models = pre_trained_models

    # Determine teaching strategy based on cluster
    if len(X_integrated.shape) == 1 or X_integrated.shape[0] == 1:  # Single student
        cluster_label = cluster_labels[0]
        is_anomaly = is_anomaly[0]
    else:
        # For multiple students, return arrays
        cluster_label = cluster_labels
        is_anomaly = is_anomaly

    teaching_strategy = assign_teaching_strategy(cluster_labels)

    review_flag = flag_students_for_review(is_anomaly, student_id)

    return {
        "student_id": student_id,
        "cluster": cluster_label,
        "is_anomaly": is_anomaly,
        "teaching_strategy": teaching_strategy,
        "review_flag": review_flag,
        "models": models
    }


def assign_teaching_strategy(cluster):
    """
    Adapt teaching strategy based on cluster assignment.

    Parameters:
    cluster (int): Cluster ID.

    Returns:
    str: Recommended teaching strategy.
    """
    def strategy_for(c):
        if c == 0:
            return "Focus on foundational skills and regular check-ins."
        elif c == 1:
            return "Encourage advanced projects and peer collaboration."
        else:
            return "Provide additional resources and interactive activities."

    if isinstance(cluster, (list, np.ndarray)):
        return [strategy_for(c) for c in cluster]
    else:
        return strategy_for(cluster)


def flag_students_for_review(is_anomaly, student_id):
    """
    Flag students for review if they are anomalies.

    Parameters:
    is_anomaly (bool): Whether the student is an anomaly.
    student_id (str): Student ID.

    Returns:
    str: Review flag message or None if no review needed.
    """
    if isinstance(is_anomaly, (list, np.ndarray)):
        flags = []
        for i, anomaly in enumerate(is_anomaly):
            if anomaly:
                sid = student_id[i] if isinstance(student_id, (list, np.ndarray)) else f"{student_id}_{i}"
                flags.append(f"Student {sid} flagged for review due to anomalous behavior.")
        return flags if flags else None
    else:
        if is_anomaly:
            return f"Student {student_id} flagged for review due to anomalous behavior."
        return None
