import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

from preprocessing import get_preprocessing_pipeline


def get_model_and_pipeline(model_name, scale_pos_weight=None):
    """
    Returns the appropriate model, pipeline, and param grid.
    """
    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)
        pipeline = Pipeline([
            ("preprocessing", get_preprocessing_pipeline(model_type="logreg")),
            ("classifier", model)
        ])
        param_grid = {}

    elif model_name == "xgb":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        pipeline = Pipeline([
            ("preprocessing", get_preprocessing_pipeline(model_type="xgb")),
            ("classifier", model)
        ])
        # Define the hyperparameter search space for the XGBoost classifier inside the pipeline
        param_grid = {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 6, 10],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
        }

    else:
        raise ValueError("Model must be 'logreg' or 'xgb'.")

    return pipeline, param_grid

# function to evaluate and log the model
def evaluate_and_log(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob)
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # cm = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    # ax.set_title("Confusion Matrix")
    # ax.set_xlabel("Predicted")
    # ax.set_ylabel("Actual")
    # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
    #     fig.savefig(tmpfile.name)
    #     mlflow.log_artifact(tmpfile.name, artifact_path="confusion_matrix")
    # plt.close(fig)


def main(model_name):
    mlflow.set_experiment("main_pipeline")

    with mlflow.start_run(run_name=f"{model_name}_full_pipeline"):

        # Load data
        df = pd.read_csv("../data/bank_data.csv")
        X = df.drop(columns=['target'])
        y = df['target'].map({"no": 0, "yes": 1})

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        #classs imbalance handling
        scale_pos_weight = None
        if model_name == "xgb":
            neg, pos = np.bincount(y_train)
            scale_pos_weight = neg / pos
            mlflow.log_param("scale_pos_weight", scale_pos_weight)

        pipeline, param_grid = get_model_and_pipeline(
            model_name=model_name,
            scale_pos_weight=scale_pos_weight
        )

        # Hyperparameter search
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=20,
            scoring="average_precision",  # AUC-PR
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        # Fit the randomized search on the training data
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)

        # Evaluation
        evaluate_and_log(best_model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(best_model, f"{model_name}_pipeline_model")

# Arguments to train the models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["logreg", "xgb"], required=True,
                        help="Which model to train: 'logreg' or 'xgb'")
    args = parser.parse_args()

    main(args.model)
