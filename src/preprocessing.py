import sys, os
sys.path.append(os.path.abspath(".."))
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


# Feature engineering function
def engineer_features(df):
    df = df.copy()
    
    # Map binary
    df["contact_mode"] = df["contact_mode"].map({"telephone": 0, "cellular": 1})
    # Since there's only 3 "yes" values, we ignore them and just consider "unknown" and "no"
    df["is_credit_unknown"] = (df["has_credit"] == "unknown").astype(int)
    
    # New engineered features
    df["any_loan"] = ((df["housing_loan"] == "yes") | (df["personal_loan"] == "yes")).astype(int)
    df["has_been_contacted_before"] = (df["nb_previous_contact"] > 0).astype(int)
    df["no_recent_contact"] = ((df["N_last_days"] == 999) & (df['nb_previous_contact'] == 0)).astype(int)
    df["recent_unsuccessful_contacts"] = ((df["previous_outcome"] == "failure") & (df["N_last_days"] < 30)).astype(int)
    
    # return with redundant features dropped.
    # Note!: dropping last_contact_duration, since it is a leaky feature
    df = df.drop(columns=['has_credit', 'last_contact_duration'])
    return df


# Preprocessing pipeline function
def get_preprocessing_pipeline(model_type=None, drop_features=None):
    
    """
    1) engineer_features
    2) drop any columns in drop_features
    3) one-hot categorical + conditional scaling on numeric
    """
    
    # Feature list
    onehot_features = [
    "occupation", "marital_status", "month", "week_day",
    "housing_loan", "personal_loan", "education", "previous_outcome"
    ]

    base_numeric_features = [
    "age", "N_last_days", "nb_previous_contact",
    "emp_var_rate", "cons_price_index", "cons_conf_index", "euri_3_month", "nb_employees"
    ]

    engineered_numeric = [
    "contact_mode", "is_credit_unknown", "any_loan", "has_been_contacted_before",
    "no_recent_contact", "recent_unsuccessful_contacts"
    ]

    # list of all numeric features
    all_numeric = base_numeric_features + engineered_numeric
    
    # drop the redundant features if needed
    drop_features = set(drop_features or [])

    # wrap engineering function
    engineer = FunctionTransformer(engineer_features, validate=False)

    # filter out any columns to drop
    onehot_feats = [f for f in onehot_features if f not in drop_features]
    numeric_feats = [f for f in all_numeric if f not in drop_features]

    # pick scaler vs passthrough
    numeric_transformer = (
        StandardScaler() if model_type == "logreg"
        else "passthrough"
    )

    # create the column transformer module to encode features
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_feats),
            ("num",    numeric_transformer,                                      numeric_feats),
        ],
        remainder="drop"
    )

    # return the full pipeline
    return Pipeline(steps=[
        ("engineer",    engineer),
        ("preprocessor", preprocessor),
    ])
