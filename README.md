# fraud_mlops

An example of how to apply MLOps principles to a predictive modelling porject. This repository uses the [Fraud Detection Transactions Dataset](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset) from Kaggle. The dataset is saved for convenience in the `data` folder as `synthetic_fraud_dataset.csv`.

Feel free to open this code using Codespaces by clicking the button below:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=954237137)

## Notes

This project uses `poetry` as a package manager and `uv` as an installer. It requires a python version of `3.10` or greater.

## EDA

Exploratory Data Analysis (EDA) is performed in the `notebooks` folder. The important feature engineering steps have been added to functions in `script/utils` to be used in the model training later.

## Model training

The model training script are in the `scripts` folder as `model_train.py`. You can either run using:

```bash
python -m scripts/model_train.py`
```

Or use the `bash` file in the root directory using:

```bash
venv_fraud_mlops/bin/python scripts/model_train.py
```

The `bash` file will also spin up the MLFlow Tracking Server too automatically. To do this manually, run:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### Model config

The model config file can be found at `scripts/model_config.py`. You can change these values and they will then be used in the model training script. The parameters will then be logged to the MLFlow Tracking Server too.

## Model Evaluation

As part of the model training above two validation datasets will be saved to `data`: `X_val` and `y_val`. These are datasets not included in the test-train approach and therefore can be used to evaluate your best model after training and tuning. You can run using:

```bash
python -m scripts/model_eval.py`
```

Or use the `bash` file in the root directory using:

```bash
venv_fraud_mlops/bin/python scripts/model_eval.py
```

This will print the evaluation scores to the terminal.

## Model Deployment

TBC. Will likely use FastAPI to deploy a model.
