# model_config.py
MODEL_PARAMS = {
    "LogisticRegression": {
        "C": 0.01,
        "solver": "saga",
        "class_weight": "balanced",
        "random_state": 101
    },
    "SGDClassifier": {
        "class_weight": "balanced",
        "learning_rate": "adaptive",
        "eta0": 0.03,
        "loss": "log_loss",
        "random_state": 101
    },
    "KNeighborsClassifier": {},
    "RandomForestClassifier": {
        "n_estimators": 50,
        "max_depth": 7,
        "class_weight": "balanced",
        "random_state": 101
    },
    "AdaBoostClassifier": {
        "n_estimators": 50,
        "learning_rate": 0.3,
        "random_state": 101
    },
    "GradientBoostingClassifier": {
        "n_estimators": 50,
        "max_depth": 7,
        "random_state": 101
    },
    "XGBClassifier": {
        "n_estimators": 50,
        "max_depth": 7,
        "enable_categorical": True,
        "random_state": 101
    }
}
