import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import streamlit as st
from utils.clustering import cluster_students


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


def evaluate_classification_model(model, X_train, y_train, X_test, y_test, model_name, task):
    try:
        # Cross-validation on training set only
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', error_score='raise')
        print(f"{task} - {model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Train on training set
        model.fit(X_train, y_train)
        # Predict on test set
        y_pred = model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"{task} - {model_name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{task} - {model_name} Classification Report:\n", classification_report(y_test, y_pred))

        return model, test_accuracy
    except Exception as e:
        print(f"{task} - {model_name} failed: {str(e)}")
        return None, -1


def evaluate_regression_model(model, X_train, y_train, X_test, y_test, model_name, task):
    try:
        # Cross-validation on training set only
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', error_score='raise')
        print(f"{task} - {model_name} Cross-Validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Train on training set
        model.fit(X_train, y_train)
        # Predict on test set
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{task} - {model_name} Test MSE: {mse:.4f}")
        print(f"{task} - {model_name} Test R²: {r2:.4f}")

        return model, r2
    except Exception as e:
        print(f"{task} - {model_name} failed: {str(e)}")
        return None, -1


@st.cache_data
def train_and_evaluate_models(_X_integrated, _X_analytics, _data):
    """
    Train and evaluate models for dropout risk, performance, and engagement prediction.
    Cache the results to avoid retraining if inputs haven't changed.
    Load existing models if available.
    Save predictions to CSV files for test set only.
    Return the updated data DataFrame with test set predictions, along with train/test indices.
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

    # Split data into train and test sets upfront
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.002, random_state=42
    )
    data_train = data.iloc[train_idx].copy()
    data_test = data.iloc[test_idx].copy()
    X_integrated_train = X_integrated[train_idx]
    X_integrated_test = X_integrated[test_idx]
    X_analytics_train = X_analytics[train_idx]
    X_analytics_test = X_analytics[test_idx]

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

        # Compute predictions for test set only
        data_test["dropout_pred_int"] = best_drop_model_int.predict(X_integrated_test)
        data_test["dropout_pred_ana"] = best_drop_model_ana.predict(X_analytics_test)
        data_test["performance_pred"] = best_perf_model.predict(X_integrated_test)

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
                current_quantile -= quantile_step
            elif low_count < min_samples:
                current_quantile += quantile_step
            current_quantile = max(0.1, min(0.9, current_quantile))
            threshold = polarity_sum.quantile(current_quantile)
            high_count = (polarity_sum > threshold).sum()
            low_count = (polarity_sum <= threshold).sum()
            print(
                f"Adjusted threshold to {current_quantile * 100}th percentile: {threshold}, High: {high_count}, Low: {low_count}"
            )

        data["engagement_label"] = polarity_sum.apply(lambda x: "High" if x > threshold else "Low")
        data_test["engagement_label"] = data["engagement_label"].iloc[test_idx]
        print("Engagement label distribution before encoding:")
        print(data["engagement_label"].value_counts())

        y_eng = le.fit_transform(data["engagement_label"])
        y_eng_test = y_eng[test_idx]
        print("Encoded y_eng distribution (test set):")
        print(pd.Series(y_eng_test).value_counts())

        # Verify that we have two classes
        unique_classes_eng = np.unique(y_eng_test)
        if len(unique_classes_eng) < 2:
            print(
                f"Warning: Engagement (Integrated) - Test set target variable 'engagement_label' has only one unique class ({unique_classes_eng}). Assigning default predictions."
            )
            data_test["engagement_pred"] = np.array(["Low"] * len(data_test))
        else:
            data_test["engagement_pred"] = le.inverse_transform(best_eng_model.predict(X_integrated_test))

        # Compute groups (clustering) on full dataset for visualization
        model, groups = cluster_students(X_integrated)

        # Compute performance metrics using test set
        # Dropout (Integrated)
        y_drop_test = data_test["dropout"]
        integrated_acc = accuracy_score(y_drop_test, data_test["dropout_pred_int"])
        integrated_report = classification_report(y_drop_test, data_test["dropout_pred_int"])
        integrated_cv = cross_val_score(best_drop_model_int, X_integrated_train, data_train["dropout"], cv=5).mean()

        # Dropout (Analytics-Only)
        analytics_acc = accuracy_score(y_drop_test, data_test["dropout_pred_ana"])
        analytics_report = classification_report(y_drop_test, data_test["dropout_pred_ana"])
        analytics_cv = cross_val_score(best_drop_model_ana, X_analytics_train, data_train["dropout"], cv=5).mean()

        print("Loaded models and computed test set predictions successfully.")
    else:
        print("Training new models...")
        # --- 1. Predict Dropout Risk (Integrated: With Sentiment) ---
        y_drop = data["dropout"]
        y_drop_train = y_drop.iloc[train_idx]
        y_drop_test = y_drop.iloc[test_idx]
        unique_classes = np.unique(y_drop_train)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Dropout Risk (Integrated) - Training set target variable 'dropout' has only one unique class ({unique_classes}). Binary classification requires at least two classes."
            )

        check_data(X_integrated_train, y_drop_train, "Dropout Risk (Integrated)")
        check_data(X_integrated_test, y_drop_test, "Dropout Risk (Integrated)")

        dropout_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }

        best_drop_model_int, best_drop_score_int, best_drop_name_int = None, -1, ""
        for name, model in dropout_models.items():
            trained_model, score = evaluate_classification_model(
                model, X_integrated_train, y_drop_train, X_integrated_test, y_drop_test, name, "Dropout Risk (Integrated)"
            )
            if trained_model and score > best_drop_score_int:
                best_drop_model_int, best_drop_score_int, best_drop_name_int = trained_model, score, name

        if best_drop_model_int:
            print(
                f"Best Dropout Risk Model (Integrated): {best_drop_name_int} with Test Accuracy: {best_drop_score_int:.4f}"
            )
            data_test["dropout_pred_int"] = best_drop_model_int.predict(X_integrated_test)
            with open(dropout_model_int_path, "wb") as f:
                pickle.dump(best_drop_model_int, f)
        else:
            raise ValueError("No valid dropout risk model (integrated) found.")

        # --- 2. Predict Student Performance (Integrated) ---
        y_perf = data["Total_Score"]
        y_perf_train = y_perf.iloc[train_idx]
        y_perf_test = y_perf.iloc[test_idx]
        check_data(X_integrated_train, y_perf_train, "Performance (Integrated)")
        check_data(X_integrated_test, y_perf_test, "Performance (Integrated)")

        performance_models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        }

        best_perf_model, best_perf_score, best_perf_name = None, -1, ""
        for name, model in performance_models.items():
            trained_model, score = evaluate_regression_model(
                model, X_integrated_train, y_perf_train, X_integrated_test, y_perf_test, name, "Performance (Integrated)"
            )
            if trained_model and score > best_perf_score:
                best_perf_model, best_perf_score, best_perf_name = trained_model, score, name

        if best_perf_model:
            print(f"Best Performance Model (Integrated): {best_perf_name} with Test R²: {best_perf_score:.4f}")
            data_test["performance_pred"] = best_perf_model.predict(X_integrated_test)
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
                current_quantile -= quantile_step
            elif low_count < min_samples:
                current_quantile += quantile_step
            current_quantile = max(0.1, min(0.9, current_quantile))
            threshold = polarity_sum.quantile(current_quantile)
            high_count = (polarity_sum > threshold).sum()
            low_count = (polarity_sum <= threshold).sum()
            print(
                f"Adjusted threshold to {current_quantile * 100}th percentile: {threshold}, High: {high_count}, Low: {low_count}"
            )

        data["engagement_label"] = polarity_sum.apply(lambda x: "High" if x > threshold else "Low")
        data_test["engagement_label"] = data["engagement_label"].iloc[test_idx]
        print("Engagement label distribution before encoding:")
        print(data["engagement_label"].value_counts())

        y_eng = le.fit_transform(data["engagement_label"])
        y_eng_train = y_eng[train_idx]
        y_eng_test = y_eng[test_idx]
        print("Encoded y_eng distribution (test set):")
        print(pd.Series(y_eng_test).value_counts())

        unique_classes_eng = np.unique(y_eng_train)
        if len(unique_classes_eng) < 2:
            print(
                f"Warning: Engagement (Integrated) - Training set target variable 'engagement_label' has only one unique class ({unique_classes_eng}). Assigning default predictions."
            )
            data_test["engagement_pred"] = np.array(["Low"] * len(data_test))
        else:
            check_data(X_integrated_train, y_eng_train, "Engagement (Integrated)")
            check_data(X_integrated_test, y_eng_test, "Engagement (Integrated)")

            engagement_models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            }

            best_eng_model, best_eng_score, best_eng_name = None, -1, ""
            for name, model in engagement_models.items():
                trained_model, score = evaluate_classification_model(
                    model, X_integrated_train, y_eng_train, X_integrated_test, y_eng_test, name, "Engagement (Integrated)"
                )
                if trained_model and score > best_eng_score:
                    best_eng_model, best_eng_score, best_eng_name = trained_model, score, name

            if best_eng_model:
                print(f"Best Engagement Model (Integrated): {best_eng_name} with Test Accuracy: {best_eng_score:.4f}")
                data_test["engagement_pred"] = le.inverse_transform(best_eng_model.predict(X_integrated_test))
                with open(engagement_model_path, "wb") as f:
                    pickle.dump(best_eng_model, f)
            else:
                raise ValueError("No valid engagement model found.")

        # --- 4. Identify Student Groups (Integrated) ---
        # Clustering on full dataset for visualization
        groups = cluster_students(X_integrated)

        # --- 5. Predict Dropout Risk (Analytics-Only) ---
        check_data(X_analytics_train, y_drop_train, "Dropout Risk (Analytics-Only)")
        check_data(X_analytics_test, y_drop_test, "Dropout Risk (Analytics-Only)")

        dropout_models_ana = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }

        best_drop_model_ana, best_drop_score_ana, best_drop_name_ana = None, -1, ""
        for name, model in dropout_models_ana.items():
            trained_model, score = evaluate_classification_model(
                model, X_analytics_train, y_drop_train, X_analytics_test, y_drop_test, name, "Dropout Risk (Analytics-Only)"
            )
            if trained_model and score > best_drop_score_ana:
                best_drop_model_ana, best_drop_score_ana, best_drop_name_ana = trained_model, score, name

        if best_drop_model_ana:
            print(
                f"Best Dropout Risk Model (Analytics-Only): {best_drop_name_ana} with Test Accuracy: {best_drop_score_ana:.4f}"
            )
            data_test["dropout_pred_ana"] = best_drop_model_ana.predict(X_analytics_test)
            with open(dropout_model_ana_path, "wb") as f:
                pickle.dump(best_drop_model_ana, f)
        else:
            raise ValueError("No valid dropout risk model (analytics-only) found.")

        # --- Model Comparison (Dropout Prediction: Integrated vs Analytics-Only) ---
        integrated_acc = best_drop_score_int
        integrated_report = classification_report(y_drop_test, data_test["dropout_pred_int"])
        integrated_cv = cross_val_score(best_drop_model_int, X_integrated_train, y_drop_train, cv=5).mean()

        analytics_acc = best_drop_score_ana
        analytics_report = classification_report(y_drop_test, data_test["dropout_pred_ana"])
        analytics_cv = cross_val_score(best_drop_model_ana, X_analytics_train, y_drop_train, cv=5).mean()

    # Save predictions to CSV files for test set
    # Integrated predictions
    data_test_filtered = data_test.drop(columns=["dropout_pred_ana"], errors='ignore')
    data_test_filtered.to_csv(integrated_predictions_path, index=False)
    print(f"Integrated test set predictions saved to {integrated_predictions_path}")

    # Analytics-Only predictions
    data_test_filtered = data_test.drop(columns=["dropout_pred_int"], errors='ignore')
    data_test_filtered.to_csv(analytics_predictions_path, index=False)
    print(f"Analytics-Only test set predictions saved to {analytics_predictions_path}")

    # Update full DataFrame with test set predictions
    data.loc[test_idx, "dropout_pred_int"] = data_test["dropout_pred_int"]
    data.loc[test_idx, "dropout_pred_ana"] = data_test["dropout_pred_ana"]
    data.loc[test_idx, "performance_pred"] = data_test["performance_pred"]
    data.loc[test_idx, "engagement_pred"] = data_test["engagement_pred"]
    data.loc[test_idx, "engagement_label"] = data_test["engagement_label"]

    # Return the updated data DataFrame along with other outputs, including test_idx
    return (
        best_drop_model_int,
        best_perf_model,
        best_eng_model,
        groups,
        integrated_acc,
        analytics_acc,
        integrated_report,
        analytics_report,
        integrated_cv,
        analytics_cv,
        data,
        test_idx
    )