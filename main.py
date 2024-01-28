# Import Libraries
import pandas as pd
import numpy as np


import yaml

# read yaml file
with open("config.yaml") as file:
    config = yaml.safe_load(file)

import processing.preprocessing as pp
import pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from functions import calculate_roc_auc
from sklearn.ensemble import RandomForestClassifier
from predict import make_prediction
import pickle


def run_training():
    """Train the model"""

    # Read Data
    train = pd.read_csv("titanic.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(columns=config["TARGET"]),
        train[config["TARGET"]],
        test_size=0.2,
        random_state=config["SEED"],
        stratify=train[config["TARGET"]],
    )

    completed_pl = Pipeline(
        steps=[
            ("preprocessor", pipeline.pipe),
            ("classifier", RandomForestClassifier()),
        ]
    )
    randomforest = completed_pl.fit(X_train, y_train)
    pickle.dump(randomforest, open("randomforest.pkl", "wb"))

    print(f"Train ROC-AUC: {calculate_roc_auc(randomforest, X_train, y_train):.4f}")
    print(f"Test ROC-AUC: {calculate_roc_auc(randomforest, X_test, y_test):.4f}")


if __name__ == "__main__":
    # run_training()
    # Test Prediction
    test_data = pd.read_csv("titanic.csv")
    make_prediction(test_data)
    print("Prediction Done")
