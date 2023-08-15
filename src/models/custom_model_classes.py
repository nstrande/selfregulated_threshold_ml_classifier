from src import config

import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from pandera.typing import Series
from typing import Union


def get_precision_threshold(
    label: Series[int],
    predict_proba: Series[float],
    precision_target: float = config.PRECISION_TARGET,
) -> float:
    """Returns the precision threshold value by which the precision target is achieved

    Args:
        label (pd.Series[int]): pandas series with labels
        predict_proba (pd.Series[float]): pandas series with prediction probability from model
        precision_target (float): precision target value by which the output precision precision target value will be achieve
    """
    precisions, _, thresholds = precision_recall_curve(label, predict_proba)
    return thresholds[np.argmax(precisions >= precision_target)]


class SelfDefinedThresholdWrapper:
    """Class to self regulated threshold with respect to the precision target"""

    def __init__(
        self, model: object, precision_target: float = config.PRECISION_TARGET
    ):
        self.model = model
        self._precision_target = precision_target

    def fit(self, X, y):
        self.model.fit(X, y)
        predict_proba = self.model.predict_proba(X)[:, 1]
        # Get model threshold based on predefined precision target
        self._threshold_precision = get_precision_threshold(y, predict_proba)
        return self

    def predict(
        self, X: pd.DataFrame, threshold_precision: Union[float, None] = None
    ) -> Series[int]:
        # Use default value if none threshold precision input
        threshold_precision = (
            self.threshold_precision
            if threshold_precision is None
            else threshold_precision
        )
        predict_proba = self.model.predict_proba(X)[:, 1]  # Positive predictions
        return predict_proba >= threshold_precision

    def predict_proba(self, X: pd.DataFrame) -> Series[float]:
        return self.model.predict_proba(X)

    @property
    def precision_target(self):
        return self._precision_target

    @property
    def threshold_precision(self):
        return self._threshold_precision
