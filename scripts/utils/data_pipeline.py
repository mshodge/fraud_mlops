import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging

# logging.basicConfig(level=logging.DEBUG)

# import warnings

# warnings.filterwarnings('ignore')


def rename_columns(df):
    """Renames dataset columns to meaningful names."""
    df.columns = [
        "transaction_id",
        "user_id",
        "transaction_amount",
        "transaction_method",
        "transaction_timestamp",
        "account_balance",
        "device_type",
        "transaction_location",
        "merchant_category",
        "ip_address_flag",
        "previous_fraudulent_activities",
        "daily_transaction_count",
        "avg_transaction_amount_7d",
        "failed_transaction_count_7d",
        "card_type",
        "card_age_months",
        "transaction_distance",
        "authentication_method",
        "fraud_risk_score",
        "is_weekend",
        "fraud_label",
    ]
    return df


def preprocess_time_features(df):
    """Extracts time-based features from the transaction timestamp."""
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    df["day_of_week"] = df["transaction_timestamp"].dt.day_name()
    df["transaction_hour"] = df["transaction_timestamp"].dt.hour
    df = df.drop(columns=["transaction_timestamp"])
    return df


def preprocess_value_features(df):
    """Creates new transaction-based features."""
    df["transaction_amount_to_balance_ratio"] = (
        df["transaction_amount"] / df["account_balance"]
    )
    df["is_high_value_transaction"] = df["transaction_amount"] > df[
        "transaction_amount"
    ].quantile(0.75)
    return df


def preprocess_card_age(df):
    """Binning card age into categories."""
    df["card_age_years"] = pd.cut(
        df["card_age_months"],
        bins=[0, 12, 60, 120, 250],
        labels=["0-1", "1-5", "5-10", "10+"],
    )
    return df.drop(columns=["card_age_months"])


def split_data(df):
    """Splits dataset into train, test, and validation sets."""
    df = df.drop(columns=["transaction_id", "user_id"])
    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_test, y_train, y_test, X_val, y_val, X, y


def encode_and_scale(X_train, X_test, X_val):
    """Encodes categorical features and scales numerical features."""
    categorical_cols = [
        "transaction_method",
        "device_type",
        "transaction_location",
        "merchant_category",
        "card_type",
        "authentication_method",
        "day_of_week",
        "card_age_years",
    ]
    encoder = OrdinalEncoder()
    scaler = StandardScaler()

    for X in [X_train, X_test, X_val]:
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    joblib.dump(scaler, "artefacts/scaler.save")
    joblib.dump(encoder, "artefacts/encoder.save")

    return X_train, X_test, X_val


def select_best_features(X, X_train, y_train, X_test, X_val, print_scores=True):
    """Selects top k best features based on ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=15)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    X_val = selector.transform(X_val)

    joblib.dump(selector, "artefacts/selector.save")

    if print_scores:
        # Checking the importance scores of each feature
        feat_scores = pd.Series(selector.scores_, index=X.columns).sort_values(
            ascending=False
        )
        print(feat_scores)

    return X_train, X_test, X_val


def data_pipeline(df, print_scores=True):
    df = rename_columns(df)
    df = preprocess_time_features(df)
    df = preprocess_value_features(df)
    df = preprocess_card_age(df)

    X_train, X_test, y_train, y_test, X_val, y_val, X, y = split_data(df)
    X_train, X_test, X_val = encode_and_scale(X_train, X_test, X_val)
    X_train, X_test, X_val = select_best_features(
        X, X_train, y_train, X_test, X_val, print_scores
    )
    return X_train, X_test, y_train, y_test, X_val, y_val, X, y


def load_encode_and_scale_and_select(X, scaler, encoder, selector):
    """Loads the encoded categorical features and scales numerical features."""
    categorical_cols = [
        "transaction_method",
        "device_type",
        "transaction_location",
        "merchant_category",
        "card_type",
        "authentication_method",
        "day_of_week",
        "card_age_years",
    ]

    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    X = scaler.transform(X)

    X = selector.transform(X)

    return X


def data_pipeline_predict(df, scaler, encoder, selector):
    df = rename_columns(df)
    df = preprocess_time_features(df)
    df = preprocess_value_features(df)
    df = preprocess_card_age(df)
    df = df.drop(columns=["transaction_id", "user_id"])
    df = df.drop(columns=["fraud_label"])
    df = load_encode_and_scale_and_select(df, scaler, encoder, selector)
    return df
