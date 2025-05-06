from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from typing import Literal

model_run_id = 'e80a771915d04b14873e966eceaa2e37'

# Load the MLflow pipeline model (adjust run ID or path as needed)
# model = mlflow.sklearn.load_model(f"runs:/{model_run_id}/xgb_pipeline_model")
model = mlflow.sklearn.load_model("./mlruns/882110903822705047/e80a771915d04b14873e966eceaa2e37/artifacts/xgb_pipeline_model")

app = FastAPI(
    title="Term Deposit Subscription Predictor",
    description="Predicts whether a customer will subscribe to a term deposit using a trained ML model pipeline.",
    version="1.0"
)

# Define the expected input schema
class CustomerData(BaseModel):
    age: int
    occupation: str
    marital_status: str
    education: str
    has_credit: str  # "yes", "no", "unknown"
    contact_mode: Literal["telephone", "cellular"]
    month: str
    week_day: str
    last_contact_duration: int
    N_last_days: int
    nb_previous_contact: int
    previous_outcome: str
    housing_loan: str  # "yes" or "no"
    personal_loan: str  # "yes" or "no"
    emp_var_rate: float
    cons_price_index: float
    cons_conf_index: float
    euri_3_month: float
    nb_employees: float

@app.get("/")
def health_check():
    return {"status": "online", "message": "Model is ready to make predictions."}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # The MLflow pipeline model handles preprocessing internally
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }
