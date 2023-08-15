from sklearn.datasets import load_breast_cancer
import pandas as pd


def fetch_data() -> pd.DataFrame:
    """
    Fetches breast cancer data as a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing breast cancer data and target labels.
    """
    data = load_breast_cancer(as_frame=True)
    dataframe = data["data"]
    dataframe["target"] = data["target"].replace(
        {0: 1, 1: 0}
    )  # Swap to demonstrate the use of selfregulated precision wrapper
    return dataframe
