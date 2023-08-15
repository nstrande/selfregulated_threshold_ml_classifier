from sklearn.model_selection import train_test_split

import pandas as pd

import typer


def main(raw_data_inpath: str, train_data_outpath: str, test_data_outpath: str) -> None:
    raw_df = pd.read_parquet(raw_data_inpath)
    print(f"Load raw data from {raw_data_inpath}")

    train_df, test_df = train_test_split(
        raw_df, test_size=0.3, stratify=raw_df["target"], random_state=123
    )  # Stratifying by 'target' column helps maintain class distribution in both training and testing sets.

    train_df.to_parquet(train_data_outpath)
    print(f"Save train data to {train_data_outpath}")

    test_df.to_parquet(test_data_outpath)
    print(f"Save test data to {test_data_outpath}")


if __name__ == "__main__":
    typer.run(main)
