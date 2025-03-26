import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import numpy as np

from utils.data_pipeline import data_pipeline
from model_config import MODEL_PARAMS

import warnings

warnings.filterwarnings('ignore')

mlflow.autolog()

def load_data(file_path: str):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains multiple models and logs performance metrics in MLflow."""
    model_classes = {
        "XGBClassifier": XGBClassifier,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier
    }

    models = {name: model_classes[name](**params) for name, params in MODEL_PARAMS.items()}

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Training and Testing CV Scores
            training_cv_score = cross_val_score(model, X_train, y_train, cv=skf, scoring = "roc_auc")
            testing_cv_score = cross_val_score(model, X_test, y_test, cv=skf, scoring = "roc_auc")
    
            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP,
                "Avg Training CV Score" : np.mean(training_cv_score),
                "Avg Testing CV Score" : np.mean(testing_cv_score),
                "Trainning CV Score Std" : np.std(training_cv_score),
                "Testing CV Score Std" : np.std(testing_cv_score)     
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(model, name)
            print(f"Logged {name} model")

def main():
    """Loads data, runs data pipeline and then evaluates models and 
    saves to mlflow."""
    df = load_data("data/synthetic_fraud_dataset.csv")

    X_train, X_test, y_train, y_test, X_val, y_val, X, y = data_pipeline(df, print_scores=True)

    # Save data sets
    pd.DataFrame(X_train).to_csv("data/X_train.csv")
    pd.DataFrame(X_test).to_csv("data/X_test.csv")
    pd.DataFrame(y_train).to_csv("data/y_train.csv")
    pd.DataFrame(y_test).to_csv("data/y_test.csv")
    pd.DataFrame(X_val).to_csv("data/X_val.csv")
    pd.DataFrame(y_val).to_csv("data/y_val.csv")

    mlflow.set_experiment("Fraud Detection Experiment")
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
