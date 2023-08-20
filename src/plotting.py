import numpy as np

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_curve

from pandera.typing import Series
from typing import Union


def plot_confusion_matrix(
    y_true: Series[int],
    y_pred: Series[int],
    normalize: bool = False,
    title: str = None,
    cmap=plt.cm.Blues,
):
    """
    Plot the confusion matrix and return the Matplotlib figure.

    Args:
        y_true (Series[int]): Ground truth labels.
        y_pred (Series[int]): Predicted labels.
        normalize (bool, optional): Whether to normalize the matrix. Default is False.
        title (str, optional): Title of the plot. Default is None.
        cmap (matplotlib colormap, optional): Color map for the plot. Default is plt.cm.Blues.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure containing the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 4))  # Just adjust the figure size as needed
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = [0, 1]
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return plt


def plot_precision_recall_curve(
    y: Union[np.ndarray, Series[int]], y_proba: Union[np.ndarray, Series[int]]
) -> plt.figure:
    """
    Plot the precision-recall curve based on predicted probabilities.

    Args:
        y (array-like): Ground truth binary labels.
        y_proba (array-like): Predicted probabilities for the positive class.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure containing the precision-recall curve plot.
    """
    precisions, recalls, threshold = precision_recall_curve(y, y_proba)

    plt.clf()
    plt.plot(threshold, precisions[:-1], "b--", label="Precision")
    plt.plot(threshold, recalls[:-1], "g--", label="Recall")
    plt.ylabel("%-score")
    plt.xlabel("Predicted probabilities")
    plt.legend(["Precision", "Recall"])

    return plt.gcf()

