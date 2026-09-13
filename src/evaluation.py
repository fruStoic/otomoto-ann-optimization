"""
Evaluation utilities for churn classification.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_probabilities(
    model,
    X,
) -> np.ndarray:
    """
    Return churn probabilities as a one-dimensional array.
    """
    probabilities = model.predict(
        X,
        verbose=0,
    )

    return probabilities.reshape(-1)


def evaluate_predictions(
    y_true,
    probabilities,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate churn predictions at a specified probability threshold.

    Churn = 1 is the positive class.
    """

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }