📖 Project Explanation
This project is a complete Credit Risk Modeling Pipeline developed as part of a machine learning assignment. The goal is to build a system that can identify high-risk customers based on their transaction history using classification models.

The workflow is structured into the following key steps:

🔹 1. Data Preparation
Raw transaction data is cleaned and enriched.

Features such as Recency, Frequency, and Monetary are calculated (RFM Analysis).

Risk labels are assigned using KMeans clustering, creating a proxy target for supervised modeling.

🔹 2. Preprocessing Pipeline
Categorical features are one-hot encoded.

Numerical features are standardized.

All transformations are wrapped into a ColumnTransformer pipeline for reusability and consistency.

🔹 3. Model Training & Experiment Tracking
Models trained: Logistic Regression and Random Forest

Each model is evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

GridSearchCV is used to perform hyperparameter tuning.

MLflow is used to:

Track experiments and metrics

Log and version models

Register the best-performing model

🔹 4. Model Deployment (API)
A FastAPI app is built to expose a /predict endpoint.

It loads the best model from the MLflow Model Registry.

Accepts a new customer's data and returns a risk probability score.

The app is containerized using Docker and easily runnable via docker-compose.

🔹 5. Continuous Integration (CI)
A GitHub Actions pipeline is configured to run on every push:

flake8 checks for code quality

pytest runs unit tests

The pipeline ensures clean, tested, and maintainable code.

🔹 6. Unit Testing
Core data processing functions are tested using pytest.

Tests ensure that data loading, splitting, and transformations work as expected.

This project demonstrates not just model building, but real-world machine learning workflow design, including:

Data versioning

Model tracking

Reproducibility

API integration

Automation via CI/CD
