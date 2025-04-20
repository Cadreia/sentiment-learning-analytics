import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


def determine_optimal_clusters(X, max_k=10):
    """Determine the optimal number of clusters using silhouette score."""
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (K): {optimal_k} with silhouette score: {max(silhouette_scores):.4f}")
    return optimal_k


def cluster_students(X):
    optimal_k = determine_optimal_clusters(X)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters


def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return iso_forest, anomalies == -1
