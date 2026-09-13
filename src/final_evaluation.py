"""
Final evaluation of the selected SGD ANN.

Optimizer selection was completed using validation data only.

Final configuration:
- Optimizer: SGD
- Learning rate: 0.01
- Seed: 42
- Architecture: Dense(32) -> Dropout(0.30) -> Dense(16) -> Sigmoid
- Loss: binary_crossentropy
- Threshold: 0.50

The held-out test set is evaluated once in this script.
"""

from pathlib import Path
import random
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_loader import (
    load_teleconnect_data,
)

from preprocessing import (
    prepare_teleconnect_data,
)

from model import (
    build_ann,
    create_early_stopping,
)

from evaluation import (
    predict_probabilities,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

SEED = 42
MAX_EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 10
THRESHOLD = 0.50


def set_seed(
    seed: int,
):
    """
    Set all relevant random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)

    tf.keras.utils.set_random_seed(
        seed
    )


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
):
    """
    Wilson confidence interval for a binomial proportion.
    """
    if total <= 0:
        raise ValueError(
            "total must be greater than zero."
        )

    alpha = 1 - confidence

    z = NormalDist().inv_cdf(
        1 - alpha / 2
    )

    p = successes / total

    denominator = (
        1
        + z**2 / total
    )

    centre = (
        p
        + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return (
        max(
            0.0,
            centre - margin,
        ),
        min(
            1.0,
            centre + margin,
        ),
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tf.keras.backend.clear_session()

    set_seed(
        SEED
    )

    df = load_teleconnect_data()

    prepared = (
        prepare_teleconnect_data(
            df,
            random_state=42,
        )
    )

    print(
        "\nFINAL MODEL CONFIGURATION"
    )

    print(
        "=" * 60
    )

    print(
        "Selected optimizer: SGD"
    )

    print(
        "Learning rate: 0.01"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Train samples: "
        f"{len(prepared.y_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(prepared.y_val)}"
    )

    print(
        f"Test samples: "
        f"{len(prepared.y_test)}"
    )

    model = build_ann(
        input_features=(
            prepared.X_train.shape[1]
        ),
        optimizer_name="sgd",
        dropout_rate=0.30,
    )

    early_stopping = (
        create_early_stopping(
            patience=PATIENCE
        )
    )

    history = model.fit(
        prepared.X_train,
        prepared.y_train,

        validation_data=(
            prepared.X_val,
            prepared.y_val,
        ),

        epochs=MAX_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=1,
    )

    val_loss = np.asarray(
        history.history["val_loss"]
    )

    best_epoch = int(
        np.argmin(val_loss) + 1
    )

    epochs_run = len(
        history.history["loss"]
    )

    print(
        "\nTRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Epochs run: "
        f"{epochs_run}"
    )

    print(
        f"Best validation-loss epoch: "
        f"{best_epoch}"
    )

    # ---------------------------------------------------------
    # FINAL TEST SET EVALUATION
    # ---------------------------------------------------------

    probabilities = (
        predict_probabilities(
            model,
            prepared.X_test,
        )
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        prepared.y_test,
        predictions,
    )

    precision = precision_score(
        prepared.y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        prepared.y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        prepared.y_test,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        prepared.y_test,
        probabilities,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            prepared.y_test,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        "=" * 60
    )

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
        "\nCONFUSION MATRIX COUNTS"
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

    # ---------------------------------------------------------
    # Majority baseline
    # ---------------------------------------------------------

    majority_accuracy = max(
        np.mean(
            prepared.y_test == 0
        ),
        np.mean(
            prepared.y_test == 1
        ),
    )

    print(
        "\nMAJORITY-CLASS BASELINE"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy: "
        f"{majority_accuracy:.3f}"
    )

    print(
        f"ANN improvement: "
        f"{accuracy - majority_accuracy:+.3f}"
    )

    # ---------------------------------------------------------
    # Confidence intervals
    # ---------------------------------------------------------

    accuracy_ci = (
        wilson_interval(
            successes=(
                int(tp)
                + int(tn)
            ),
            total=len(
                prepared.y_test
            ),
        )
    )

    recall_ci = (
        wilson_interval(
            successes=int(tp),
            total=(
                int(tp)
                + int(fn)
            ),
        )
    )

    precision_ci = (
        wilson_interval(
            successes=int(tp),
            total=(
                int(tp)
                + int(fp)
            ),
        )
    )

    print(
        "\n95% WILSON CONFIDENCE INTERVALS"
    )

    print(
        "=" * 60
    )

    print(
        "Accuracy: "
        f"{accuracy_ci[0]:.3f} "
        f"to "
        f"{accuracy_ci[1]:.3f}"
    )

    print(
        "Precision: "
        f"{precision_ci[0]:.3f} "
        f"to "
        f"{precision_ci[1]:.3f}"
    )

    print(
        "Recall: "
        f"{recall_ci[0]:.3f} "
        f"to "
        f"{recall_ci[1]:.3f}"
    )

    # ---------------------------------------------------------
    # Save final metrics
    # ---------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "final_metrics.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FINAL SGD TEST RESULTS\n"
        )

        file.write(
            "=" * 50
            + "\n"
        )

        file.write(
            f"seed: {SEED}\n"
        )

        file.write(
            f"threshold: "
            f"{THRESHOLD}\n"
        )

        file.write(
            f"epochs_run: "
            f"{epochs_run}\n"
        )

        file.write(
            f"best_epoch: "
            f"{best_epoch}\n"
        )

        file.write(
            f"accuracy: "
            f"{accuracy:.6f}\n"
        )

        file.write(
            f"precision: "
            f"{precision:.6f}\n"
        )

        file.write(
            f"recall: "
            f"{recall:.6f}\n"
        )

        file.write(
            f"f1: "
            f"{f1:.6f}\n"
        )

        file.write(
            f"roc_auc: "
            f"{auc:.6f}\n"
        )

        file.write(
            f"tn: {tn}\n"
        )

        file.write(
            f"fp: {fp}\n"
        )

        file.write(
            f"fn: {fn}\n"
        )

        file.write(
            f"tp: {tp}\n"
        )

        file.write(
            "\n95% Wilson CI\n"
        )

        file.write(
            "accuracy: "
            f"{accuracy_ci[0]:.6f}, "
            f"{accuracy_ci[1]:.6f}\n"
        )

        file.write(
            "precision: "
            f"{precision_ci[0]:.6f}, "
            f"{precision_ci[1]:.6f}\n"
        )

        file.write(
            "recall: "
            f"{recall_ci[0]:.6f}, "
            f"{recall_ci[1]:.6f}\n"
        )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    matrix = np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix
    )

    plt.xticks(
        [0, 1],
        [
            "No churn",
            "Churn",
        ],
    )

    plt.yticks(
        [0, 1],
        [
            "No churn",
            "Churn",
        ],
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "Actual class"
    )

    plt.title(
        "Final SGD Confusion Matrix"
    )

    for row in range(2):
        for column in range(2):

            plt.text(
                column,
                row,
                matrix[
                    row,
                    column,
                ],
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        prepared.y_test,
        probabilities,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"SGD ANN "
            f"(AUC = {auc:.3f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel(
        "False positive rate"
    )

    plt.ylabel(
        "True positive rate"
    )

    plt.title(
        "Final SGD ANN ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nFinal outputs saved."
    )

    print(
        "\nDO NOT RETUNE THE MODEL "
        "BASED ON THESE TEST RESULTS."
    )


if __name__ == "__main__":
    main()