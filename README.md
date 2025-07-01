📖 Project Explanation
This project implements a full Credit Risk Modeling Pipeline that detects potentially high-risk customers based on historical transaction data. It covers every stage of a real-world ML lifecycle—from preprocessing and model selection to deployment and CI/CD.

🚀 Project Highlights
🔹 1. Data Preparation
Cleaned raw transaction data.

Extracted RFM features (Recency, Frequency, Monetary).

Used KMeans clustering to generate a proxy risk label (is_high_risk) for supervised training.

🔹 2. Preprocessing Pipeline
Built using ColumnTransformer:

Categorical variables → One-Hot Encoding.

Numerical variables → Standard Scaler.

Ensures robust and consistent input transformations across the pipeline.

🔹 3. Model Training & Experiment Tracking
Models Used:

Logistic Regression

Random Forest Classifier

Hyperparameter Tuning:

Used GridSearchCV with cross-validation to tune:

Logistic Regression (C, penalty)

Random Forest (n_estimators, max_depth)

Evaluation Metrics Tracked via MLflow:

Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
LogisticRegression	0.9998	1.0000	0.9974	0.9987	1.0000
RandomForest	0.9998	1.0000	0.9974	0.9987	1.0000

Model Management:

Models and metrics are tracked in MLflow

The best model is registered in the MLflow Model Registry under BestCreditModel

🌐 API Deployment with FastAPI
Developed an API using FastAPI:

Endpoint: POST /predict

Accepts customer features in JSON format

Returns a risk_probability score

The API:

Loads the best model from the MLflow Model Registry

Is containerized using Docker

Can be run locally using docker-compose up --build

🧪 Unit Testing
Includes tests for data loading and processing using pytest

Ensures that helper functions behave correctly and robustly

🔄 CI/CD with GitHub Actions
GitHub Actions workflow (.github/workflows/ci.yml) automates:

Code style check using flake8

Running all unit tests with pytest

The build fails if linting or tests fail, promoting clean and tested code

📁 Technologies Used
Python, Pandas, Scikit-learn

MLflow (Tracking + Registry)

FastAPI

Docker & Docker Compose

GitHub Actions (CI)

Pytest & Flake8

💡 How to Run Locally
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/credit-risk-model
cd credit-risk-model
Build and Run the API:

bash
Copy
Edit
docker-compose up --build
Access API at:

bash
Copy
Edit
http://localhost:8000/docs
Access MLflow UI:

bash
Copy
Edit
mlflow ui
Visit http://localhost:5000
