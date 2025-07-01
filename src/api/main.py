import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, RiskPredictionResponse

app = FastAPI()

# 📌 Use direct path to model artifact instead of model registry
# Replace this with the correct Run ID from MLflow UI
RUN_ID = "2d42581361cf42bf85e087a2ebd55e8b"  # 👈 Replace this!
MODEL_PATH = f"mlruns/0/{RUN_ID}/artifacts/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(customer: CustomerData):
    # Convert Pydantic input to DataFrame
    data = pd.DataFrame([customer.dict()])

    # Predict probability of risk (positive class)
    pred_prob = model.predict(data).astype(float)[0]

    return RiskPredictionResponse(risk_probability=pred_prob)
