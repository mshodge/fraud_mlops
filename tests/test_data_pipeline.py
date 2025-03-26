import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """Create a sample dataset for testing"""
        data = {
            "transaction_id": range(1, 6),
            "user_id": range(101, 106),
            "transaction_amount": [100, 200, 50, 300, 400],
            "transaction_method": [
                "Credit Card",
                "Debit Card",
                "Credit Card",
                "Debit Card",
                "PayPal",
            ],
            "transaction_timestamp": pd.date_range(
                start="2023-01-01", periods=5, freq="D"
            ),
            "account_balance": [1000, 1500, 1200, 800, 2000],
            "device_type": ["Mobile", "Desktop", "Tablet", "Mobile", "Desktop"],
            "transaction_location": ["NY", "CA", "TX", "NY", "CA"],
            "merchant_category": ["Retail", "Grocery", "Retail", "Grocery", "Retail"],
            "ip_address_flag": [0, 1, 0, 1, 0],
            "previous_fraudulent_activities": [1, 0, 2, 1, 3],
            "daily_transaction_count": [3, 1, 2, 4, 5],
            "avg_transaction_amount_7d": [150, 180, 90, 250, 300],
            "failed_transaction_count_7d": [0, 2, 1, 0, 3],
            "card_type": ["Visa", "Mastercard", "Visa", "Mastercard", "Amex"],
            "card_age_months": [10, 24, 60, 120, 240],
            "transaction_distance": [5, 10, 15, 20, 25],
            "authentication_method": [
                "PIN",
                "Password",
                "PIN",
                "Password",
                "Biometric",
            ],
            "fraud_risk_score": [0.1, 0.5, 0.3, 0.9, 0.2],
            "is_weekend": [0, 1, 0, 1, 0],
            "fraud_label": [0, 1, 0, 1, 0],
        }
        self.df = pd.DataFrame(data)

    def test_rename_columns(self):
        from scripts.utils.data_pipeline import rename_columns

        df_renamed = rename_columns(self.df.copy())

        expected_columns = [
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
        self.assertEqual(list(df_renamed.columns), expected_columns)

    def test_preprocess_time_features(self):
        from scripts.utils.data_pipeline import preprocess_time_features

        df_transformed = preprocess_time_features(self.df.copy())
        self.assertIn("day_of_week", df_transformed.columns)
        self.assertIn("transaction_hour", df_transformed.columns)
        self.assertNotIn("transaction_timestamp", df_transformed.columns)

    def test_preprocess_value_features(self):
        from scripts.utils.data_pipeline import preprocess_value_features

        df_transformed = preprocess_value_features(self.df.copy())
        self.assertIn("transaction_amount_to_balance_ratio", df_transformed.columns)
        self.assertIn("is_high_value_transaction", df_transformed.columns)

    def test_preprocess_card_age(self):
        from scripts.utils.data_pipeline import preprocess_card_age

        df_transformed = preprocess_card_age(self.df.copy())
        self.assertIn("card_age_years", df_transformed.columns)
        self.assertNotIn("card_age_months", df_transformed.columns)

    def test_split_data(self):
        from scripts.utils.data_pipeline import split_data

        X_train, X_test, y_train, y_test, X_val, y_val, X, y = split_data(
            self.df.copy()
        )
        self.assertEqual(len(X_train) + len(X_test) + len(X_val), len(self.df))
        self.assertEqual(len(y_train) + len(y_test) + len(y_val), len(self.df))

    def test_encode_and_scale(self):
        from scripts.utils.data_pipeline import (
            preprocess_time_features,
            preprocess_value_features,
            preprocess_card_age,
        )
        from scripts.utils.data_pipeline import encode_and_scale
        from scripts.utils.data_pipeline import split_data

        df = preprocess_time_features(self.df.copy())
        df = preprocess_value_features(df)
        df = preprocess_card_age(df)
        X_train, X_test, y_train, y_test, X_val, y_val, X, y = split_data(df)
        X_train, X_test, X_val = encode_and_scale(X_train, X_test, X_val)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.shape[1], X_val.shape[1])

    def test_select_best_features(self):
        from scripts.utils.data_pipeline import (
            preprocess_time_features,
            preprocess_value_features,
        )
        from scripts.utils.data_pipeline import preprocess_card_age
        from scripts.utils.data_pipeline import select_best_features
        from scripts.utils.data_pipeline import split_data, encode_and_scale

        df = preprocess_time_features(self.df.copy())
        df = preprocess_value_features(df)
        df = preprocess_card_age(df)
        X_train, X_test, y_train, y_test, X_val, y_val, X, y = split_data(df)
        X_train, X_test, X_val = encode_and_scale(X_train, X_test, X_val)
        X_train, X_test, X_val = select_best_features(
            X, X_train, y_train, X_test, X_val, print_scores=False
        )
        self.assertEqual(X_train.shape[1], 15)
        self.assertEqual(X_test.shape[1], 15)
        self.assertEqual(X_val.shape[1], 15)

    def test_data_pipeline(self):
        from scripts.utils.data_pipeline import data_pipeline

        X_train, X_test, y_train, y_test, X_val, y_val, X, y = data_pipeline(
            self.df.copy(), print_scores=False
        )
        self.assertEqual(X_train.shape[1], 15)
        self.assertEqual(X_test.shape[1], 15)
        self.assertEqual(X_val.shape[1], 15)
        self.assertEqual(len(X_train) + len(X_test) + len(X_val), len(self.df))


if __name__ == "__main__":
    unittest.main()
