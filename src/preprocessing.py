"""
Leakage-safe preprocessing for the Teleconnect churn dataset.

Improvements over the supplied notebook:
- preserves all 7,043 customers;
- resolves blank TotalCharges values explicitly;
- one-hot encodes categorical predictors;
- standardises numeric predictors;
- creates separate train, validation, and test sets;
- fits all preprocessing using training data only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    preprocessor: ColumnTransformer
    feature_names: list[str]

    train_rows: int
    val_rows: int
    test_rows: int


def clean_teleconnect_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the raw Teleconnect dataset.

    The 11 blank TotalCharges values all occur for customers
    with tenure == 0. They are treated as zero accumulated
    charges rather than deleting those customers.
    """
    if df is None or df.empty:
        raise ValueError(
            "Input dataset cannot be empty."
        )

    cleaned = df.copy()

    required = {
        ID_COLUMN,
        TARGET,
        "TotalCharges",
    }

    missing = required.difference(
        cleaned.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    # Convert blank strings to NaN, then numeric.
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"]
        .replace(r"^\s*$", np.nan, regex=True),
        errors="coerce",
    )

    missing_total_charges = (
        cleaned["TotalCharges"].isna()
    )

    # Verify the known blank-value pattern.
    invalid_missing = (
        missing_total_charges
        & (cleaned["tenure"] != 0)
    )

    if invalid_missing.any():
        raise ValueError(
            "TotalCharges contains missing values "
            "for customers with tenure greater than 0."
        )

    # A tenure-zero customer has not accumulated prior charges.
    cleaned.loc[
        missing_total_charges,
        "TotalCharges",
    ] = 0.0

    # customerID is an identifier, not a predictor.
    cleaned = cleaned.drop(
        columns=[ID_COLUMN]
    )

    if cleaned.isna().any().any():
        raise ValueError(
            "Missing values remain after cleaning."
        )

    return cleaned


def encode_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors and encode:
        Churn Yes -> 1
        Churn No  -> 0
    """
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    observed = set(
        df[TARGET].unique()
    )

    expected = {"Yes", "No"}

    if not observed.issubset(expected):
        raise ValueError(
            f"Unexpected Churn labels: "
            f"{sorted(observed)}"
        )

    X = df.drop(
        columns=[TARGET]
    ).copy()

    y = df[TARGET].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    return X, y


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build preprocessing using column roles found in training data.

    Numeric:
        StandardScaler

    Categorical:
        OneHotEncoder, dropping the first level to reduce
        redundant dummy columns.
    """
    numeric_features = [
        column
        for column in NUMERIC_FEATURES
        if column in X_train.columns
    ]

    categorical_features = [
        column
        for column in X_train.columns
        if column not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> PreparedData:
    """
    Create a 70/15/15 stratified split.

    Preprocessing is fitted ONLY on the training split.
    """

    # 70% train, 30% temporary holdout.
    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Divide temporary holdout equally:
    # 15% validation, 15% test.
    (
        X_val,
        X_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    return PreparedData(
        X_train=np.asarray(
            X_train_processed,
            dtype=np.float32,
        ),
        X_val=np.asarray(
            X_val_processed,
            dtype=np.float32,
        ),
        X_test=np.asarray(
            X_test_processed,
            dtype=np.float32,
        ),

        y_train=y_train.to_numpy(
            dtype=np.int32
        ),
        y_val=y_val.to_numpy(
            dtype=np.int32
        ),
        y_test=y_test.to_numpy(
            dtype=np.int32
        ),

        preprocessor=preprocessor,
        feature_names=feature_names,

        train_rows=len(X_train),
        val_rows=len(X_val),
        test_rows=len(X_test),
    )


def prepare_teleconnect_data(
    df: pd.DataFrame,
    random_state: int = 42,
) -> PreparedData:
    """
    Run the complete preprocessing pipeline.
    """
    cleaned = clean_teleconnect_data(
        df
    )

    X, y = encode_target(
        cleaned
    )

    return split_and_preprocess(
        X,
        y,
        random_state=random_state,
    )