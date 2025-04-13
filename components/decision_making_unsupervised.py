# components/decision_making_unsupervised.py
def decision_making_unsupervised(student_data, clusters, anomalies):
    actions = []
    for i, (cluster, anomaly) in enumerate(zip(clusters, anomalies)):
        student_id = i
        strategy = f"Adapt Teaching Strategy for Cluster {cluster}: "
        if cluster == 0:
            strategy += "Increase interactive content"
        elif cluster == 1:
            strategy += "Provide more practice problems"
        else:
            strategy += "Focus on conceptual understanding"
        actions.append(strategy)

        if anomaly:
            action = f"Flag Student {student_id} for Review (Detected as Anomaly)"
            actions.append(action)

    return actions
