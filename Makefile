install_dev:
	pip install -r requirements.txt
	pip install -e .

run_mlflow_app:
	mlflow ui

# Steps
get_raw_data:
	python src/steps/get_raw_data.py data/raw_data.parquet

get_train_test_data:
	python src/steps/get_train_test_data.py data/raw_data.parquet data/train_data.parquet data/test_data.parquet

train_random_forest:
	python src/steps/train.py data/train_data.parquet data/test_data.parquet src.models.model_configs.random_forest

train_xgb:
	python src/steps/train.py data/train_data.parquet data/test_data.parquet src.models.model_configs.xgboost

train_decision_tree:
	python src/steps/train.py data/train_data.parquet data/test_data.parquet src.models.model_configs.decision_tree

train_logistic_regression:
	python src/steps/train.py data/train_data.parquet data/test_data.parquet src.models.model_configs.logistic_regression

train_selfregulated_threshold_random_forest:
	python src/steps/train.py data/train_data.parquet data/test_data.parquet src.models.model_configs.selfregulated_threshold_random_forest


# Pipeline
train_pipeline: get_raw_data get_train_test_data train_selfregulated_threshold_random_forest
