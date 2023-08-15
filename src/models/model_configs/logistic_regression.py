import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import ColumnSelector

from sklearn.base import ClassifierMixin

from src import config
from typing import Dict, Union


RUN_NAME = "logistic_regression"


# Define pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("scaler", StandardScaler())]),
            config.NUMERIC_FEATURES,
        )
    ]
)


def model() -> ClassifierMixin:
    """
    Create a logistic regression classifer pipeline.

    Returns:
        ClassifierMixin: A logistic regression classifer pipeline.
    """
    # Create a RandomForestClassifier model
    decision_tree_model = LogisticRegression()

    # Create a pipeline for preprocessing and classification
    pipeline = Pipeline(
        [
            ("column_selector", ColumnSelector(config.ALL_FEATURES)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                decision_tree_model,
            ),
        ]
    )
    return pipeline
