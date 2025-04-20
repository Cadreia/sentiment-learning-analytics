from scripts.modeling import cluster_students, detect_anomalies


def learning_path_unsupervised(X):
    clusters = cluster_students(X)
    anomalies = detect_anomalies(X)
    return clusters, anomalies
