from src.data.fetch_data import fetch_data

import typer

from pathlib import Path


def main(outpath: str) -> None:
    df = fetch_data()
    print("Raw data fetched")

    df.to_parquet(Path(outpath))
    print(f"Raw data saved here: {outpath}")


if __name__ == "__main__":
    typer.run(main)
