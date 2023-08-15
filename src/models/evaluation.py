import inspect
from typing import Tuple, Dict
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src import config


class BaseEvaluate:
    def __init__(self) -> None:
        pass

    def _get_metrics_dict(self) -> Dict[str, float]:
        """
        Returns a dictionary of metrics computed by this object.

        This method inspects all non-private methods of the class instance, excluding those that are
        properties, and calls them to obtain a name and a metric value. The name is used as the key in
        the resulting dictionary, and the metric value is the corresponding value.

        Returns:
            A dictionary with string keys and float values, where each key represents a metric name
            and each value represents the corresponding metric value.
        """
        metrics_dict = {}
        for method, _ in inspect.getmembers(self, predicate=inspect.ismethod):
            if not isinstance(
                getattr(type(self), method, None), property
            ) and not method.startswith("_"):
                name, metric = getattr(type(self), method, None)(self)
                metrics_dict[name] = metric
        return metrics_dict

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics_dict


class Evaluate(BaseEvaluate):
    """
    Attributes:
        metrics (Dict[str, float]): A dictionary containing the names and values of all the computed metrics.

    Note:
        To include additional evaluation metrics in this class, simply define new methods following
        the pattern of existing metric methods with Tuple[str, float] as return. They will automatically be included in the '_get_metrics_dict' method.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._y = dataframe[config.TARGET_COLUMN]
        self._y_hat = dataframe["predict"]
        self._y_hat_proba = dataframe["predict_proba"]
        self._metrics_dict = self._get_metrics_dict()

    def get_f1_score(self) -> Tuple[str, float]:
        return "f1_score", f1_score(self._y, self._y_hat)

    def get_accuracy_score(self) -> Tuple[str, float]:
        return "accuracy_score", accuracy_score(self._y, self._y_hat)

    def get_precision_score(self) -> Tuple[str, float]:
        return "precision_score", precision_score(self._y, self._y_hat)

    def get_recall_score(self) -> Tuple[str, float]:
        return "recall_score", recall_score(self._y, self._y_hat)
