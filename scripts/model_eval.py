import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow.sklearn


X_val = pd.read_csv('data/X_val.csv', index_col=0)
# y_val = pd.read_csv('data/X_val.csv', index_col=0)
y_val = pd.read_csv('data/y_val.csv', index_col=0).values

logged_model = 'runs:/ad71048a0b904cae82d8209ba71d06cb/XGBClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
y_pred = loaded_model.predict(X_val)

# y_pred_proba = loaded_model.predict_proba(X_val)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1_score': f1_score(y_val, y_pred)
    }

print(metrics)