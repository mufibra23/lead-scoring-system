"""
Model Training + SHAP Explainability for Lead Scoring
=====================================================
Trains Logistic Regression (baseline) and XGBoost,
evaluates both, computes SHAP values, and saves everything.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import shap

# Add parent dir to path so we can import src
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_and_clean, engineer_features, prepare_for_modeling


def prepare_data():
    """Load, clean, engineer, split."""
    df = load_and_clean()
    df = engineer_features(df)
    X, y = prepare_for_modeling(df)
    
    # Split: 80/20 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples ({y_train.mean():.2%} converted)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.mean():.2%} converted)")
    
    return X_train, X_test, y_train, y_test


def prepare_for_logistic(X_train, X_test):
    """Label-encode categoricals for Logistic Regression (can't handle category dtype)."""
    X_train_lr = X_train.copy()
    X_test_lr = X_test.copy()
    
    cat_cols = X_train_lr.select_dtypes(include=["category"]).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined train+test to avoid unseen labels
        all_values = pd.concat([X_train_lr[col], X_test_lr[col]]).astype(str)
        le.fit(all_values)
        X_train_lr[col] = le.transform(X_train_lr[col].astype(str))
        X_test_lr[col] = le.transform(X_test_lr[col].astype(str))
        encoders[col] = le
    
    return X_train_lr, X_test_lr, encoders


def evaluate_model(name, model, X_test, y_test, y_pred, y_proba=None):
    """Print evaluation metrics for a model."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'=' * 50}")
    print(f"{name} — RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC:   {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None}


def train_baseline(X_train, X_test, y_train, y_test):
    """Train Logistic Regression as baseline."""
    X_train_lr, X_test_lr, _ = prepare_for_logistic(X_train, X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_lr, y_train)
    
    y_pred = model.predict(X_test_lr)
    y_proba = model.predict_proba(X_test_lr)[:, 1]
    
    metrics = evaluate_model("BASELINE: Logistic Regression", model, X_test_lr, y_test, y_pred, y_proba)
    return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with categorical support."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model("XGBoost", model, X_test, y_test, y_pred, y_proba)
    
    # Cross-validation
    print("Cross-validation (5-fold ROC-AUC)...")
    cv_scores = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            enable_categorical=True, tree_method="hist", random_state=42,
        ),
        X_train, y_train, cv=5, scoring="roc_auc"
    )
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    metrics["cv_roc_auc"] = cv_scores.mean()
    
    return model, metrics


def compute_shap_values(model, X_test):
    """Compute SHAP values using TreeExplainer (fast for XGBoost)."""
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Global feature importance (mean absolute SHAP)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_test.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    
    print("\nTop 10 Features by SHAP Importance:")
    for i, (feat, val) in enumerate(feature_importance.head(10).items()):
        bar = "█" * int(val / feature_importance.max() * 20)
        print(f"  {i+1:2d}. {feat:40s} {val:.4f} {bar}")
    
    return explainer, shap_values, feature_importance


def save_artifacts(model, explainer, shap_values, feature_importance, X_test, y_test, metrics):
    """Save model, SHAP values, and metadata."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("shap_cache", exist_ok=True)
    
    # Save model
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved: models/xgboost_model.pkl")
    
    # Save SHAP values
    with open("shap_cache/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)
    print("Saved: shap_cache/shap_values.pkl")
    
    # Save feature importance
    feature_importance.to_csv("shap_cache/feature_importance.csv")
    print("Saved: shap_cache/feature_importance.csv")
    
    # Save test data + predictions for the dashboard
    test_data = X_test.copy()
    test_data["actual_converted"] = y_test.values
    test_data["predicted_proba"] = model.predict_proba(X_test)[:, 1]
    test_data["predicted_converted"] = model.predict(X_test)
    test_data["lead_score"] = (test_data["predicted_proba"] * 100).round(1)
    test_data.to_csv("shap_cache/test_predictions.csv", index=False)
    print("Saved: shap_cache/test_predictions.csv")
    
    # Save metrics
    with open("shap_cache/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("Saved: shap_cache/metrics.pkl")
    
    print("\nAll artifacts saved successfully!")


def main():
    print("=" * 60)
    print("LEAD SCORING MODEL TRAINING")
    print("=" * 60)
    
    # 1. Prepare data
    print("\n--- Step 1: Prepare Data ---")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 2. Train baseline
    print("\n--- Step 2: Train Baseline (Logistic Regression) ---")
    lr_model, lr_metrics = train_baseline(X_train, X_test, y_train, y_test)
    
    # 3. Train XGBoost
    print("\n--- Step 3: Train XGBoost ---")
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    
    # 4. Compare
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Logistic':>10} {'XGBoost':>10} {'Winner':>10}")
    print("-" * 50)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lr_val = lr_metrics[metric]
        xgb_val = xgb_metrics[metric]
        winner = "XGBoost" if xgb_val > lr_val else "Logistic"
        print(f"{metric:<15} {lr_val:>10.4f} {xgb_val:>10.4f} {winner:>10}")
    
    # 5. SHAP explainability
    print("\n--- Step 4: SHAP Explainability ---")
    explainer, shap_values, feature_importance = compute_shap_values(xgb_model, X_test)
    
    # 6. Save everything
    print("\n--- Step 5: Save Artifacts ---")
    save_artifacts(xgb_model, explainer, shap_values, feature_importance, X_test, y_test, xgb_metrics)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Next step: python app.py  (run the Streamlit dashboard)")


if __name__ == "__main__":
    main()
