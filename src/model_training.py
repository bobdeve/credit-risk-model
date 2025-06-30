import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 🔧 Add project root to sys.path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_and_split_data
from src.preprocessing import get_preprocessor


def train_and_log_models(data_path):
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    # Define categorical and numerical columns
    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId',
        'ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId'
    ]
    numerical_cols = [
        col for col in X_train.columns
        if col not in categorical_cols and X_train[col].dtype in ['int64', 'float64']
    ]

    # Build the preprocessing pipeline
    preprocessor = get_preprocessor(categorical_cols, numerical_cols)

    # Define models to evaluate
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Start MLflow experiment
    mlflow.set_experiment("Credit Risk Modeling")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Fit model
            full_pipeline.fit(X_train, y_train)

            # Predict
            preds = full_pipeline.predict(X_test)
            probs = full_pipeline.predict_proba(X_test)[:, 1]

            # Evaluate
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)

            # Log parameters and metrics
            mlflow.log_param("model", name)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc
            })

            # Log model
           # mlflow.sklearn.log_model(full_pipeline, artifact_path="model")
            mlflow.sklearn.log_model(full_pipeline, name="model")


            print(f"✅ {name} | Accuracy: {acc:.4f} | ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_and_processed_risk_rfm.csv'
    ))
    train_and_log_models(data_path)
