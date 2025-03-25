source venv_fraud_mlops/bin/activate
venv_fraud_mlops/bin/python scripts/model_train.py
mlflow server --host 127.0.0.1 --port 8080