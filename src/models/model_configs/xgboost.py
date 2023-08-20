import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import ColumnSelector

from xgboost.sklearn import XGBClassifier

from sklearn.base import ClassifierMixin

from src import config
from typing import Dict, Union


RUN_NAME = "xgboost"


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
    Create a random forest classifier pipeline.

    Returns:
        ClassifierMixin: A random forest classifier pipeline.
    """
    # Create a RandomForestClassifier model
    xgb_model = XGBClassifier(n_estimators=3)

    # Create a pipeline for preprocessing and classification
    pipeline = Pipeline(
        [
            ("column_selector", ColumnSelector(config.ALL_FEATURES)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb_model,
            ),
        ]
    )
    return pipeline
