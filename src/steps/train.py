import typer
from importlib import import_module

from tempfile import TemporaryDirectory

import pandas as pd

import mlflow

from src import config
from src.models.evaluation import Evaluate

from src.utils import log_metric_to_mlflow, set_seed, add_predictions_to_dataframe
from src.plotting import plot_confusion_matrix, plot_precision_recall_curve


def main(
    train_data_inpath: str, test_data_inpath: str, model_config_module: str
) -> None:
    set_seed()

    print(f"Load train dataframe from {train_data_inpath}")
    train_df = pd.read_parquet(train_data_inpath)

    print(f"Load test dataframe from {test_data_inpath}")
    test_df = pd.read_parquet(test_data_inpath)

    print(f"Import model config module from {model_config_module}")
    model_config = import_module(model_config_module)

    print(f"Starting MLFlow run")
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_config.RUN_NAME}"):
        print(f"Train model")

        model = model_config.model()
        model.fit(train_df[config.ALL_FEATURES], train_df[config.TARGET_COLUMN])

        print("Add predictions to train dataframe")
        train_df = add_predictions_to_dataframe(model, train_df)

        print("Get train metrics")
        train_metrics = Evaluate(train_df).metrics
        print(f"Train metrics: {train_metrics}")
        log_metric_to_mlflow(train_metrics, view="train")

        print("Add predictions to test dataframe")
        test_df = add_predictions_to_dataframe(model, test_df)

        print("Get test metrics")
        test_metrics = Evaluate(test_df).metrics
        print(f"Test metrics: {test_metrics}")
        log_metric_to_mlflow(test_metrics, view="test")

        with TemporaryDirectory() as tmpdirname:
            print("Save holdout set")
            test_df.to_parquet(f"{tmpdirname}/holdout_set.parquet")
            mlflow.log_artifact(f"{tmpdirname}/holdout_set.parquet")

            print("Generate and log confusion matrix")
            confusion_matrix_plot = plot_confusion_matrix(
                y_true=test_df[config.TARGET_COLUMN], y_pred=test_df["predict"]
            )
            confusion_matrix_plot.savefig(f"{tmpdirname}/confusion_matrix_plot.jpg")
            mlflow.log_artifact(f"{tmpdirname}/confusion_matrix_plot.jpg")

            print("Generate and log precision recall curve plot")
            precision_recall_curve_plot = plot_precision_recall_curve(
                y=test_df[config.TARGET_COLUMN], y_proba=test_df["predict_proba"]
            )
            precision_recall_curve_plot.savefig(
                f"{tmpdirname}/precision_recall_curve_plot.jpg"
            )
            mlflow.log_artifact(f"{tmpdirname}/precision_recall_curve_plot.jpg")


if __name__ == "__main__":
    typer.run(main)
