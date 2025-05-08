# Bank Marketing Case – Subscription Prediction

This project addresses a case from a major bank aiming to improve the efficiency of its marketing campaigns. The goal is to train an ML model to predict whether a customer will subscribe to a financial product based on their demographic, financial, and campaign interaction data.

---
## Project Structure
```
bank-marketing-case/
├── data/
│ ├── bank_data.csv
│ └── bank_dataset_description.txt
│
├── notebooks/
│ ├── 01_exploratory_data_analysis.ipynb
│ ├── 02_preprocessing & feature engineering.ipynb
│ ├── 03_baseline_model_LR.ipynb
│ ├── 04_XGBoost_model.ipynb
│ └── 05_model_and_feature_analysis.ipynb
│
├── src/
│ ├── init.py
│ ├── api.py # FastAPI app for serving predictions
│ ├── main.py # Training pipeline and MLflow logging
│ └── preprocessing.py # Custom feature engineering and transformation
│
│
├── README.md
└── requirements.txt
```
---
## How to train the model


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model:
From the project root, first navigate to the `src` folder, then run one of the following commands:

```bash
python main.py --model logreg
```
or
```bash
python main.py --model xgb
```

This will:

Preprocess and engineer features

Perform hyperparameter tuning (RandomizedSearchCV)

Train and evaluate the model using AUC-PR

Log metrics and artifacts using MLflow

Save the best pipeline for deployment

### 3. View MLflow Results
To start the MLflow UI:

```bash
mlflow ui
```
Then open `http://127.0.0.1:5000` in your browser to explore runs, metrics, and artifacts.

---
## How to Run the API

Once you've trained and saved the model (via `main.py`), you can serve it through a REST API using **FastAPI**.
> **Note:** If you're using MLflow to load the model (`mlflow.sklearn.load_model(...)`),  
> make sure to **adjust the path or run ID** in `api.py` to match your local MLflow setup.
>  
> You can find the path in your `mlruns/` directory or copy the run ID from the MLflow UI.


### 1. Start the API Server

```bash
uvicorn api:app --reload
```
This will start a local server at:
`http://127.0.0.1:8000`

### 2. Interactive Swagger Docs
Open `http://127.0.0.1:8000/docs`
You’ll find an interactive UI to test the /predict endpoint.

### 3. Sample Request Payload
```bash
{
  "age": 35,
  "occupation": "technician",
  "marital_status": "married",
  "education": "university.degree",
  "has_credit": "no",
  "contact_mode": "cellular",
  "month": "aug",
  "week_day": "mon",
  "N_last_days": 90,
  "nb_previous_contact": 1,
  "previous_outcome": "nonexistent",
  "housing_loan": "no",
  "personal_loan": "no",
  "emp_var_rate": 1.1,
  "cons_price_index": 93.2,
  "cons_conf_index": -36.4,
  "euri_3_month": 4.96,
  "nb_employees": 5191
}
```
### 4. Sample Response
```bash
{
  "prediction": 1,
  "probability": 0.7823
}
```
---
## Notebooks Overview

The `notebooks/` directory contains the step-by-step exploratory and modeling workflow used to analyze the data, build the models, and evaluate performance.

| Notebook | Purpose |
|----------|---------|
| `01_exploratory_data_analysis.ipynb` | Exploratory analysis to uncover patterns, class imbalance, feature distributions, and early behavioral insights |
| `02_preprocessing & feature engineering.ipynb` | Prototyping and testing feature transformations and engineered variables before moving to pipeline code |
| `03_baseline_model_LR.ipynb` | Logistic Regression baseline model, focusing on interpretability and precision |
| `04_XGBoost_model.ipynb` | XGBoost training and tuning, with evaluation focused on recall and AUC-PR for the minority class |
| `05_model_and_feature_analysis.ipynb` | Comparative model evaluation, feature importance inspection, and SHAP analysis for interpretability |

These notebooks document the full lifecycle of the ML solution before moving to production-ready scripts and APIs.

---
## Disclaimer

This project was developed as part of a technical case interview with **2021.AI**.

All code, analysis, and documentation are original work by the candidate and intended solely for demonstration and evaluation purposes.

Do not reproduce or distribute without permission.

