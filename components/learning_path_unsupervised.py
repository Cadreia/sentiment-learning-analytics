# components/learning_path_unsupervised.py
from scripts.predictive_modeling import cluster_students, detect_anomalies

def learning_path_unsupervised(X):
    clusters = cluster_students(X, n_clusters=3)
    anomalies = detect_anomalies(X)
    return clusters, anomalies