# models/predictive_modeling.py
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_integrated = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_integrated.fit(X_train, y_train)
    integrated_pred = rf_integrated.predict(X_test)
    integrated_acc = accuracy_score(y_test, integrated_pred)
    integrated_report = classification_report(y_test, integrated_pred)

    X_train_no_sentiment = X_train[:, :-2]  # Exclude sentiment features
    X_test_no_sentiment = X_test[:, :-2]
    rf_analytics = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_analytics.fit(X_train_no_sentiment, y_train)
    analytics_pred = rf_analytics.predict(X_test_no_sentiment)
    analytics_acc = accuracy_score(y_test, analytics_pred)
    analytics_report = classification_report(y_test, analytics_pred)

    integrated_cv = cross_val_score(rf_integrated, X, y, cv=5).mean()
    analytics_cv = cross_val_score(rf_analytics, X[:, :-2], y, cv=5).mean()

    return (rf_integrated, rf_analytics, integrated_acc, analytics_acc,
            integrated_report, analytics_report, integrated_cv, analytics_cv)


def cluster_students(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters


def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies == -1


def detect_outliers(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X)
    return outliers == -1
