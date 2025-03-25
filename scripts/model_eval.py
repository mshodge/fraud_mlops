import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


X_val = pd.read_csv('data/X_val.csv', index_col=0)
# y_val = pd.read_csv('data/X_val.csv', index_col=0)

y_val = np.loadtxt(open("data/y_val.csv", "rb"), delimiter=",", skiprows=1)


logged_model = 'runs:/613ccf5a62fa4affab596d5177589989/XGBClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



# Predict on a Pandas DataFrame.
y_pred = loaded_model.predict(pd.DataFrame(X_val))

# y_pred_proba = loaded_model.predict_proba(X_val)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1_score': f1_score(y_val, y_pred)
    }

print(metrics)