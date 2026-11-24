import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]
BOOLEAN_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["log_amount_usd"] = np.log1p(frame["amount_usd"].clip(lower=0))
    frame["log_hours_since_last_txn"] = np.log1p(
        frame["hours_since_last_txn"].clip(lower=0)
    )
    frame["log_distance_from_home_km"] = np.log1p(
        frame["distance_from_home_km"].clip(lower=0)
    )
    frame["log_account_balance_usd"] = np.log1p(
        frame["account_balance_usd"].clip(lower=0)
    )
    frame["amount_to_balance"] = frame["amount_usd"] / (
        frame["account_balance_usd"].abs() + 1.0
    )
    frame["risk_flag_count"] = frame[BOOLEAN_COLUMNS].astype(np.float32).sum(axis=1)
    frame["is_night"] = (
        (frame["time_of_day_hour"] >= 22) | (frame["time_of_day_hour"] <= 5)
    ).astype(np.float32)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["time_of_day_hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["time_of_day_hour"] / 24)
    frame["day_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["day_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    return frame


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_feature_columns(features)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("boolean", "passthrough", BOOLEAN_COLUMNS),
        ],
        remainder="drop",
    )


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in {"id", "is_fraud"}
    ]


def transform_features(preprocessor, train, validation, test):
    train_array = preprocessor.fit_transform(train).astype(np.float32)
    validation_array = preprocessor.transform(validation).astype(np.float32)
    test_array = preprocessor.transform(test).astype(np.float32)
    return train_array, validation_array, test_array
