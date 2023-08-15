import pandas as pd
import numpy as np

import mlflow
from typing import Dict, Literal

from sklearn.base import ClassifierMixin


def set_seed(seed=33):
    np.random.seed(seed)
    return seed


def log_metric_to_mlflow(
    metrics: Dict[str, float], view: Literal["train", "test"]
) -> None:
    """
    Log metrics to MLflow. Metrics are logged as artifacts, with the artifact name being a combination
    of the view name ("train" or "test") and the metric name.

    Args:
        metrics (Dict[str, float]): A dictionary of metric names and values to be logged.
        view (Literal["train", "test"]): The view name, indicating whether the metrics are for training
            or testing.

    Raises:
        AssertionError: Only "train" and "test" are allowed as view.
    """
    assert view in ["train", "test"], "Only train and test are allowed as view"
    for metric, value in metrics.items():
        mlflow.log_metric(f"{view}_{metric}", round(value, 2))


def add_predictions_to_dataframe(
    model: ClassifierMixin, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Add model predictions and predicted probabilities to a DataFrame.

    Args:
        model (ClassifierMixin): A trained classifier model.
        dataframe (pd.DataFrame): The DataFrame to which predictions will be added.

    Returns:
        pd.DataFrame: The input DataFrame with additional 'predict' and 'predict_proba' columns.

    Note:
        The 'predict_proba' column provides probabilities for the positive class (class 1).
    """
    dataframe["predict"] = model.predict(dataframe)
    dataframe["predict_proba"] = model.predict_proba(dataframe)[:, 1]
    return dataframe
