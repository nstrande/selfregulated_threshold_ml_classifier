import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import ColumnSelector

from sklearn.base import ClassifierMixin

from src import config
from typing import Dict, Union


RUN_NAME = "decision_tree"


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
    Create a Decision Tree classifier pipeline.

    Returns:
        ClassifierMixin: A Decision Tree classifier pipeline.
    """
    # Create a RandomForestClassifier model
    decision_tree_model = DecisionTreeClassifier()

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
