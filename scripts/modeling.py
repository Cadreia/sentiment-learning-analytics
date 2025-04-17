# models/modeling.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import pickle
import os
import streamlit as st


def check_data(X, y, task):
    """Validate input data for NaNs, infinities, and type consistency."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{task} - X must be a NumPy array, got {type(X)}")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"{task} - X contains non-numeric data: {X.dtype}")
    if np.isnan(X).any():
        raise ValueError(f"{task} - Features (X) contain NaN values.")
    if np.isinf(X).any():
        raise ValueError(f"{task} - Features (X) contain infinite values.")

    if isinstance(y, np.ndarray):
        if np.issubdtype(y.dtype, np.number):
            if np.isnan(y).any():
                raise ValueError(f"{task} - Target (y) contains NaN values.")
            if np.isinf(y).any():
                raise ValueError(f"{task} - Target (y) contains infinite values.")
        else:
            if np.any(y == None) or np.any(y == np.nan):
                raise ValueError(f"{task} - Target (y) contains None or NaN values.")
    elif isinstance(y, (pd.Series, pd.DataFrame)):
        if y.isnull().any().any():
            raise ValueError(f"{task} - Target (y) contains NaN values.")
        if np.issubdtype(y.dtype, np.number):
            if np.isinf(y).any().any():
                raise ValueError(f"{task} - Target (y) contains infinite values.")
    else:
        raise TypeError(f"{task} - Unsupported type for y: {type(y)}")

    print(f"{task} - Data check passed: No NaNs or infinities detected.")


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name, task):
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', error_score='raise')
        print(f"{task} - {model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"{task} - {model_name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{task} - {model_name} Classification Report:\n", classification_report(y_test, y_pred))

        return model, test_accuracy
    except Exception as e:
        print(f"{task} - {model_name} failed: {str(e)}")
        return None, -1


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name, task):
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', error_score='raise')
        print(f"{task} - {model_name} Cross-Validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{task} - {model_name} Test MSE: {mse:.4f}")
        print(f"{task} - {model_name} Test R²: {r2:.4f}")

        return model, r2
    except Exception as e:
        print(f"{task} - {model_name} failed: {str(e)}")
        return None, -1


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
    return clusters


def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies == -1


def detect_outliers(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X)
    return outliers == -1


@st.cache_data
def train_and_evaluate_models(_X_integrated, _X_analytics, _data):
    """
    Train and evaluate models for dropout risk, performance, and engagement prediction.
    Cache the results to avoid retraining if inputs haven't changed.
    Load existing models if available.
    Save predictions to CSV files.
    Return the updated data DataFrame along with other outputs.
    """
    # Convert inputs to hashable types for caching
    X_integrated = _X_integrated
    X_analytics = _X_analytics
    data = _data.copy()  # Create a copy to avoid modifying the original DataFrame

    # Paths to saved models and prediction files
    dropout_model_int_path = "models/dropout_model_integrated.pkl"
    dropout_model_ana_path = "models/dropout_model_analytics.pkl"
    performance_model_path = "models/performance_model.pkl"
    engagement_model_path = "models/engagement_model.pkl"
    integrated_predictions_path = "data/integrated_data_predictions.csv"
    analytics_predictions_path = "data/analytics_data_predictions.csv"

    # Check if all models exist
    models_exist = all(
        os.path.exists(path) for path in [
            dropout_model_int_path,
            dropout_model_ana_path,
            performance_model_path,
            engagement_model_path
        ]
    )

    if models_exist:
        print("Loading existing models from the 'models' folder...")
        with open(dropout_model_int_path, "rb") as f:
            best_drop_model_int = pickle.load(f)
        with open(dropout_model_ana_path, "rb") as f:
            best_drop_model_ana = pickle.load(f)
        with open(performance_model_path, "rb") as f:
            best_perf_model = pickle.load(f)
        with open(engagement_model_path, "rb") as f:
            best_eng_model = pickle.load(f)

        # Compute predictions using loaded models
        data["dropout_pred_int"] = best_drop_model_int.predict(X_integrated)
        data["dropout_pred_ana"] = best_drop_model_ana.predict(X_analytics)
        data["performance_pred"] = best_perf_model.predict(X_integrated)

        # Engagement prediction requires LabelEncoder
        le = LabelEncoder()
        polarity_sum = data["coursecontent_sentiment_score"] + data["labwork_sentiment_score"]

        # Use a percentile-based threshold to ensure both classes are represented
        threshold = polarity_sum.quantile(0.5)  # Start with the 50th percentile (median)
        print(f"Initial engagement threshold (50th percentile polarity sum): {threshold}")

        # Adjust threshold to ensure both classes are present
        high_count = (polarity_sum > threshold).sum()
        low_count = (polarity_sum <= threshold).sum()
        target_ratio = 0.1  # Ensure at least 10% of samples in the minority class
        total_samples = len(polarity_sum)
        min_samples = target_ratio * total_samples

        # If one class is underrepresented, adjust the threshold
        quantile_step = 0.05  # Adjust by 5% increments
        current_quantile = 0.5
        while high_count < min_samples or low_count < min_samples:
            if high_count < min_samples:
                # Too few "High" labels, lower the threshold
                current_quantile -= quantile_step
            elif low_count < min_samples:
                # Too few "Low" labels, raise the threshold
                current_quantile += quantile_step

            # Ensure quantile stays within valid bounds
            current_quantile = max(0.1, min(0.9, current_quantile))
            threshold = polarity_sum.quantile(current_quantile)
            high_count = (polarity_sum > threshold).sum()
            low_count = (polarity_sum <= threshold).sum()
            print(
                f"Adjusted threshold to {current_quantile * 100}th percentile: {threshold}, High: {high_count}, Low: {low_count}")

        data["engagement_label"] = polarity_sum.apply(lambda x: "High" if x > threshold else "Low")
        print("Engagement label distribution before encoding:")
        print(data["engagement_label"].value_counts())

        y_eng = le.fit_transform(data["engagement_label"])
        print("Encoded y_eng distribution:")
        print(pd.Series(y_eng).value_counts())

        # Verify that we have two classes
        unique_classes_eng = np.unique(y_eng)
        if len(unique_classes_eng) < 2:
            print(
                f"Warning: Engagement (Integrated) - Target variable 'engagement_label' has only one unique class ({unique_classes_eng}). Assigning default predictions.")
            # Assign default predictions: all "Low" (or encoded as 0)
            data["engagement_pred"] = np.array(["Low"] * len(data))
        else:
            data["engagement_pred"] = le.inverse_transform(best_eng_model.predict(X_integrated))

        # Compute groups (clustering)
        groups = cluster_students(X_integrated)

        # Compute performance metrics using a train-test split
        # Dropout (Integrated)
        y_drop = data["dropout"]
        X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_integrated, y_drop, test_size=0.2,
                                                                            random_state=42)
        integrated_acc = accuracy_score(y_test_int, best_drop_model_int.predict(X_test_int))
        integrated_report = classification_report(y_test_int, best_drop_model_int.predict(X_test_int))
        integrated_cv = cross_val_score(best_drop_model_int, X_integrated, y_drop, cv=5).mean()

        # Dropout (Analytics-Only)
        X_train_ana, X_test_ana, y_train_ana, y_test_ana = train_test_split(X_analytics, y_drop, test_size=0.2,
                                                                            random_state=42)
        analytics_acc = accuracy_score(y_test_ana, best_drop_model_ana.predict(X_test_ana))
        analytics_report = classification_report(y_test_ana, best_drop_model_ana.predict(X_test_ana))
        analytics_cv = cross_val_score(best_drop_model_ana, X_analytics, y_drop, cv=5).mean()

        print("Loaded models and computed predictions successfully.")
    else:
        print("Training new models...")
        # --- 1. Predict Dropout Risk (Integrated: With Sentiment) ---
        y_drop = data["dropout"]
        unique_classes = np.unique(y_drop)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Dropout Risk (Integrated) - Target variable 'dropout' has only one unique class ({unique_classes}). Binary classification requires at least two classes.")

        check_data(X_integrated, y_drop, "Dropout Risk (Integrated)")
        X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_integrated, y_drop, test_size=0.2,
                                                                            random_state=42)

        dropout_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            # "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            # "SVM": SVC(random_state=42, probability=True)
        }

        best_drop_model_int, best_drop_score_int, best_drop_name_int = None, -1, ""
        for name, model in dropout_models.items():
            trained_model, score = evaluate_classification_model(model, X_train_int, X_test_int, y_train_int,
                                                                 y_test_int, name, "Dropout Risk (Integrated)")
            if trained_model and score > best_drop_score_int:
                best_drop_model_int, best_drop_score_int, best_drop_name_int = trained_model, score, name

        if best_drop_model_int:
            print(
                f"Best Dropout Risk Model (Integrated): {best_drop_name_int} with Test Accuracy: {best_drop_score_int:.4f}")
            data["dropout_pred_int"] = best_drop_model_int.predict(X_integrated)
            with open(dropout_model_int_path, "wb") as f:
                pickle.dump(best_drop_model_int, f)
        else:
            raise ValueError("No valid dropout risk model (integrated) found.")

        # --- 2. Predict Student Performance (Integrated) ---
        y_perf = data["Total_Score"]
        check_data(X_integrated, y_perf, "Performance (Integrated)")
        X_train_perf, X_test_perf, y_train_perf, y_test_perf = train_test_split(X_integrated, y_perf, test_size=0.2,
                                                                                random_state=42)

        performance_models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            # "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            # "LinearRegression": LinearRegression(),
            # "SVR": SVR()
        }

        best_perf_model, best_perf_score, best_perf_name = None, -1, ""
        for name, model in performance_models.items():
            trained_model, score = evaluate_regression_model(model, X_train_perf, X_test_perf, y_train_perf,
                                                             y_test_perf, name, "Performance (Integrated)")
            if trained_model and score > best_perf_score:
                best_perf_model, best_perf_score, best_perf_name = trained_model, score, name

        if best_perf_model:
            print(f"Best Performance Model (Integrated): {best_perf_name} with Test R²: {best_perf_score:.4f}")
            data["performance_pred"] = best_perf_model.predict(X_integrated)
            with open(performance_model_path, "wb") as f:
                pickle.dump(best_perf_model, f)
        else:
            raise ValueError("No valid performance model found.")

        # --- 3. Predict Student Engagement (Integrated) ---
        le = LabelEncoder()
        polarity_sum = data["coursecontent_sentiment_score"] + data["labwork_sentiment_score"]

        # Use a percentile-based threshold to ensure both classes are represented
        threshold = polarity_sum.quantile(0.5)  # Start with the 50th percentile (median)
        print(f"Initial engagement threshold (50th percentile polarity sum): {threshold}")

        # Adjust threshold to ensure both classes are present
        high_count = (polarity_sum > threshold).sum()
        low_count = (polarity_sum <= threshold).sum()
        target_ratio = 0.1  # Ensure at least 10% of samples in the minority class
        total_samples = len(polarity_sum)
        min_samples = target_ratio * total_samples

        # If one class is underrepresented, adjust the threshold
        quantile_step = 0.05  # Adjust by 5% increments
        current_quantile = 0.5
        while high_count < min_samples or low_count < min_samples:
            if high_count < min_samples:
                # Too few "High" labels, lower the threshold
                current_quantile -= quantile_step
            elif low_count < min_samples:
                # Too few "Low" labels, raise the threshold
                current_quantile += quantile_step

            # Ensure quantile stays within valid bounds
            current_quantile = max(0.1, min(0.9, current_quantile))
            threshold = polarity_sum.quantile(current_quantile)
            high_count = (polarity_sum > threshold).sum()
            low_count = (polarity_sum <= threshold).sum()
            print(
                f"Adjusted threshold to {current_quantile * 100}th percentile: {threshold}, High: {high_count}, Low: {low_count}")

        data["engagement_label"] = polarity_sum.apply(lambda x: "High" if x > threshold else "Low")
        print("Engagement label distribution before encoding:")
        print(data["engagement_label"].value_counts())

        y_eng = le.fit_transform(data["engagement_label"])
        print("Encoded y_eng distribution:")
        print(pd.Series(y_eng).value_counts())

        unique_classes_eng = np.unique(y_eng)
        if len(unique_classes_eng) < 2:
            print(
                f"Warning: Engagement (Integrated) - Target variable 'engagement_label' has only one unique class ({unique_classes_eng}). Assigning default predictions.")
            # Assign default predictions: all "Low"
            data["engagement_pred"] = np.array(["Low"] * len(data))
        else:
            check_data(X_integrated, y_eng, "Engagement (Integrated)")
            X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_integrated, y_eng, test_size=0.2,
                                                                                random_state=42)

            engagement_models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                # "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                # "SVM": SVC(random_state=42)
            }

            best_eng_model, best_eng_score, best_eng_name = None, -1, ""
            for name, model in engagement_models.items():
                trained_model, score = evaluate_classification_model(model, X_train_eng, X_test_eng, y_train_eng,
                                                                     y_test_eng, name, "Engagement (Integrated)")
                if trained_model and score > best_eng_score:
                    best_eng_model, best_eng_score, best_eng_name = trained_model, score, name

            if best_eng_model:
                print(f"Best Engagement Model (Integrated): {best_eng_name} with Test Accuracy: {best_eng_score:.4f}")
                data["engagement_pred"] = le.inverse_transform(best_eng_model.predict(X_integrated))
                with open(engagement_model_path, "wb") as f:
                    pickle.dump(best_eng_model, f)
            else:
                raise ValueError("No valid engagement model found.")

        # --- 4. Identify Student Groups (Integrated) ---
        groups = cluster_students(X_integrated)

        # --- 5. Predict Dropout Risk (Analytics-Only) ---
        check_data(X_analytics, y_drop, "Dropout Risk (Analytics-Only)")
        X_train_ana, X_test_ana, y_train_ana, y_test_ana = train_test_split(X_analytics, y_drop, test_size=0.2,
                                                                            random_state=42)

        dropout_models_ana = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            # "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            # "SVM": SVC(random_state=42, probability=True)
        }

        best_drop_model_ana, best_drop_score_ana, best_drop_name_ana = None, -1, ""
        for name, model in dropout_models_ana.items():
            trained_model, score = evaluate_classification_model(model, X_train_ana, X_test_ana, y_train_ana,
                                                                 y_test_ana, name, "Dropout Risk (Analytics-Only)")
            if trained_model and score > best_drop_score_ana:
                best_drop_model_ana, best_drop_score_ana, best_drop_name_ana = trained_model, score, name

        if best_drop_model_ana:
            print(
                f"Best Dropout Risk Model (Analytics-Only): {best_drop_name_ana} with Test Accuracy: {best_drop_score_ana:.4f}")
            data["dropout_pred_ana"] = best_drop_model_ana.predict(X_analytics)
            with open(dropout_model_ana_path, "wb") as f:
                pickle.dump(best_drop_model_ana, f)
        else:
            raise ValueError("No valid dropout risk model (analytics-only) found.")

        # --- Model Comparison (Dropout Prediction: Integrated vs Analytics-Only) ---
        integrated_acc = best_drop_score_int
        integrated_report = classification_report(y_test_int, best_drop_model_int.predict(X_test_int))
        integrated_cv = cross_val_score(best_drop_model_int, X_integrated, y_drop, cv=5).mean()

        analytics_acc = best_drop_score_ana
        analytics_report = classification_report(y_test_ana, best_drop_model_ana.predict(X_test_ana))
        analytics_cv = cross_val_score(best_drop_model_ana, X_analytics, y_drop, cv=5).mean()

    # Save predictions to CSV files

    # Integrated predictions
    # Drop the "dropout_pred_ana" column and save the rest
    data_filtered = data.drop(columns=["dropout_pred_ana"])

    data_filtered.to_csv(integrated_predictions_path, index=False)
    print(f"Integrated predictions saved to {integrated_predictions_path}")

    # Analytics-Only predictions
    # Drop the "dropout_pred_int" column and save the rest
    data_filtered = data.drop(columns=["dropout_pred_int"])

    data_filtered.to_csv(analytics_predictions_path, index=False)
    print(f"Analytics-Only predictions saved to {analytics_predictions_path}")

    # Return the updated data DataFrame along with other outputs
    return (best_drop_model_int, best_perf_model, best_eng_model, groups,
            integrated_acc, analytics_acc, integrated_report, analytics_report,
            integrated_cv, analytics_cv, data)
