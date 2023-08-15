import pandas as pd
import numpy as np

from typing import Dict, Union
from sklearn.base import ClassifierMixin  # Used for type hint

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import ColumnSelector

from src.models.custom_model_classes import SelfDefinedThresholdWrapper
from src import config


RUN_NAME = "selfregulated_threshold_random_forest"


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
    rf_model = RandomForestClassifier(n_estimators=3)

    # Create a pipeline for preprocessing and classification
    pipeline = Pipeline(
        [
            ("column_selector", ColumnSelector(config.ALL_FEATURES)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                SelfDefinedThresholdWrapper(rf_model),
            ),
        ]
    )
    return pipeline
