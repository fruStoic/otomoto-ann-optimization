"""
Recreation of the ANN supplied in the course notebook.

This file preserves the main modelling choices of the existing model so
that its performance and weaknesses can be documented before optimization.

It is NOT the final optimized pipeline.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data_loader import load_teleconnect_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE):
    """
    Add reproducibility to the notebook baseline.

    The original notebook does not fix all random seeds, but we do so here
    so the recreated baseline can be rerun consistently.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def prepare_legacy_data(
    df: pd.DataFrame,
):
    """
    Recreate the notebook's preprocessing as closely as possible.

    Important:
    - categorical columns are label encoded;
    - blank TotalCharges rows are removed;
    - customerID is removed;
    - 75/25 stratified split is used;
    - SMOTE is applied to training data only;
    - numeric features are NOT standardized.
    """
    working = df.copy()

    categorical_features = working.drop(
        columns=[
            "customerID",
            "TotalCharges",
            "MonthlyCharges",
            "SeniorCitizen",
            "tenure",
        ]
    )

    encoder = LabelEncoder()

    encoded_categories = (
        categorical_features
        .apply(encoder.fit_transform)
    )

    retained_numeric = working[
        [
            "customerID",
            "TotalCharges",
            "MonthlyCharges",
            "SeniorCitizen",
            "tenure",
        ]
    ]

    final_dataset = pd.merge(
        retained_numeric,
        encoded_categories,
        left_index=True,
        right_index=True,
    )

    # The supplied notebook drops blank TotalCharges rows.
    blank_mask = (
        final_dataset["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    dropped_blank_rows = int(
        blank_mask.sum()
    )

    final_dataset = final_dataset.loc[
        ~blank_mask
    ].copy()

    final_dataset["TotalCharges"] = (
        pd.to_numeric(
            final_dataset["TotalCharges"],
            errors="raise",
        )
    )

    final_dataset = (
        final_dataset
        .dropna()
        .drop(columns=["customerID"])
    )

    X = final_dataset.drop(
        columns=["Churn"]
    )

    y = final_dataset["Churn"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            train_size=0.75,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_balanced, y_train_balanced = (
        smote.fit_resample(
            X_train,
            y_train,
        )
    )

    return (
        X_train_balanced,
        X_test,
        y_train_balanced,
        y_test,
        dropped_blank_rows,
    )


def build_legacy_ann(
    input_features: int,
) -> tf.keras.Model:
    """
    Recreate the ANN architecture in the course notebook.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_features,)
            ),

            tf.keras.layers.Dense(
                19,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                15,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                10,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="legacy_ann",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
        ],
    )

    return model


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed()

    df = load_teleconnect_data()

    print("\nRAW DATA")
    print("-" * 60)

    print(
        f"Shape: {df.shape}"
    )

    print("\nChurn distribution:")
    print(
        df["Churn"].value_counts()
    )

    majority_baseline = (
        df["Churn"]
        .value_counts(normalize=True)
        .max()
    )

    print(
        f"\nMajority-class baseline: "
        f"{majority_baseline:.3f}"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        dropped_blank_rows,
    ) = prepare_legacy_data(df)

    print("\nLEGACY PREPROCESSING")
    print("-" * 60)

    print(
        f"Blank TotalCharges rows dropped: "
        f"{dropped_blank_rows}"
    )

    print(
        f"Input features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Balanced training shape: "
        f"{X_train.shape}"
    )

    print(
        f"Test shape: "
        f"{X_test.shape}"
    )

    print("\nBalanced training target:")
    print(
        y_train.value_counts()
    )

    model = build_legacy_ann(
        input_features=X_train.shape[1]
    )

    print("\nLEGACY ANN")
    print("-" * 60)

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(
            X_test,
            y_test,
        ),
        verbose=1,
    )

    probabilities = (
        model.predict(
            X_test,
            verbose=0,
        )
        .reshape(-1)
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        )
        .ravel()
    )

    print("\nLEGACY BASELINE RESULTS")
    print("-" * 60)

    print(
        f"Accuracy:  {accuracy:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall:    {recall:.3f}"
    )

    print(
        f"F1-score:  {f1:.3f}"
    )

    print(
        f"ROC-AUC:   {auc:.3f}"
    )

    print(
        f"True negatives:  {tn}"
    )

    print(
        f"False positives: {fp}"
    )

    print(
        f"False negatives: {fn}"
    )

    print(
        f"True positives:  {tp}"
    )


if __name__ == "__main__":
    main()