source venv_continuum_xgboost/bin/activate
venv_continuum_xgboost/bin/python scripts/model_train.py
mlflow server --host 127.0.0.1 --port 8080