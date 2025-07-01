import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 🔧 Enable importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_and_split_data
from src.preprocessing import get_preprocessor

def train_and_log_models(data_path):
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId',
        'ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId'
    ]
    numerical_cols = [
        col for col in X_train.columns
        if col not in categorical_cols and X_train[col].dtype in ['int64', 'float64']
    ]

    preprocessor = get_preprocessor(categorical_cols, numerical_cols)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(random_state=42)
    }

    param_grids = {
        'LogisticRegression': {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__solver': ['liblinear']
        },
        'RandomForest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20]
        }
    }

    mlflow.set_experiment("Credit Risk Modeling")

    # Track best run
    best_score = 0
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            preds = best_model.predict(X_test)
            probs = best_model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)

            # Log metrics and parameters
            mlflow.log_param("model", name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc
            })

            # Log the model
            mlflow.sklearn.log_model(best_model, name="model")

            print(f"✅ {name} | Accuracy: {acc:.4f} | ROC AUC: {roc_auc:.4f}")

            # Save best model info
            if roc_auc > best_score:
                best_score = roc_auc
                best_run_id = run.info.run_id

    # ✅ Register the best model
    if best_run_id:
        mlflow.register_model(f"runs:/{best_run_id}/model", "BestCreditModel")
        print(f"🚀 Best model registered from run ID: {best_run_id}")

if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_and_processed_risk_rfm.csv'
    ))
    train_and_log_models(data_path)
