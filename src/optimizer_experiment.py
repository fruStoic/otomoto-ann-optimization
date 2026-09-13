"""
Controlled comparison of Adam, RMSprop, and SGD.

Each optimizer is evaluated across five random seeds while:
- using the same data split;
- using the same ANN architecture;
- using the same binary-cross-entropy loss;
- using the same batch size;
- using the same maximum epochs;
- using the same early-stopping rule.

Optimizer selection is based only on validation data.
The final test set is NOT used in this script.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import (
    load_teleconnect_data,
)

from preprocessing import (
    prepare_teleconnect_data,
)

from model import (
    build_ann,
    create_early_stopping,
    OPTIMIZER_CONFIGS,
)

from evaluation import (
    evaluate_predictions,
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

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]

OPTIMIZERS = [
    "adam",
    "rmsprop",
    "sgd",
]

MAX_EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 10
THRESHOLD = 0.50


def set_seed(
    seed: int,
):
    """
    Set Python, NumPy, and TensorFlow seeds.
    """
    random.seed(seed)
    np.random.seed(seed)

    tf.keras.utils.set_random_seed(
        seed
    )


def run_single_experiment(
    optimizer_name: str,
    seed: int,
    prepared,
) -> tuple[dict, pd.DataFrame]:
    """
    Train and evaluate one optimizer/seed combination.
    """

    # Clear prior TensorFlow state between runs.
    tf.keras.backend.clear_session()

    set_seed(seed)

    model = build_ann(
        input_features=(
            prepared.X_train.shape[1]
        ),
        optimizer_name=optimizer_name,
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

        verbose=0,
    )

    probabilities = (
        predict_probabilities(
            model,
            prepared.X_val,
        )
    )

    metrics = evaluate_predictions(
        y_true=prepared.y_val,
        probabilities=probabilities,
        threshold=THRESHOLD,
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

    learning_rate = (
        OPTIMIZER_CONFIGS[
            optimizer_name
        ]["learning_rate"]
    )

    result = {
        "optimizer": optimizer_name,
        "seed": seed,

        "learning_rate": (
            learning_rate
        ),

        "epochs_run": (
            epochs_run
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_val_loss": float(
            val_loss[
                best_epoch - 1
            ]
        ),

        **metrics,
    }

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        range(
            1,
            len(history_df) + 1,
        ),
    )

    history_df.insert(
        0,
        "seed",
        seed,
    )

    history_df.insert(
        0,
        "optimizer",
        optimizer_name,
    )

    return result, history_df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_teleconnect_data()

    # IMPORTANT:
    # Keep the data split fixed at random_state=42.
    # Seeds vary neural-network initialization/training,
    # not which customers belong to each split.
    prepared = prepare_teleconnect_data(
        df,
        random_state=42,
    )

    results = []
    histories = []

    print(
        "\nOPTIMIZER EXPERIMENT"
    )

    print(
        "=" * 70
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
        "Test set: NOT USED"
    )

    print(
        f"Input features: "
        f"{prepared.X_train.shape[1]}"
    )

    print(
        f"Runs planned: "
        f"{len(OPTIMIZERS) * len(SEEDS)}"
    )

    for optimizer_name in OPTIMIZERS:

        print(
            f"\n{optimizer_name.upper()}"
        )

        print(
            "-" * 70
        )

        for seed in SEEDS:

            result, history_df = (
                run_single_experiment(
                    optimizer_name=(
                        optimizer_name
                    ),
                    seed=seed,
                    prepared=prepared,
                )
            )

            results.append(
                result
            )

            histories.append(
                history_df
            )

            print(
                f"seed={seed:<4} "
                f"epochs="
                f"{result['epochs_run']:<3} "
                f"best="
                f"{result['best_epoch']:<3} "
                f"F1="
                f"{result['f1']:.3f} "
                f"Recall="
                f"{result['recall']:.3f} "
                f"Precision="
                f"{result['precision']:.3f} "
                f"AUC="
                f"{result['roc_auc']:.3f}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "experiment_results.csv",
        index=False,
    )

    histories_df = pd.concat(
        histories,
        ignore_index=True,
    )

    histories_df.to_csv(
        OUTPUT_DIR
        / "experiment_histories.csv",
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "best_val_loss",
        "epochs_run",
    ]

    summary = (
        results_df
        .groupby("optimizer")[
            metric_columns
        ]
        .agg(
            ["mean", "std"]
        )
    )

    print(
        "\nOPTIMIZER SUMMARY "
        "(MEAN ± SD ACROSS 5 SEEDS)"
    )

    print(
        "=" * 70
    )

    for optimizer_name in OPTIMIZERS:

        subset = results_df[
            results_df[
                "optimizer"
            ] == optimizer_name
        ]

        print(
            f"\n{optimizer_name.upper()}"
        )

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]:

            mean_value = (
                subset[
                    metric
                ].mean()
            )

            std_value = (
                subset[
                    metric
                ].std()
            )

            print(
                f"{metric:10s}: "
                f"{mean_value:.3f} "
                f"± {std_value:.3f}"
            )

        print(
            "epochs    : "
            f"{subset['epochs_run'].mean():.1f} "
            f"± "
            f"{subset['epochs_run'].std():.1f}"
        )

    # Flatten MultiIndex columns for CSV.
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in summary.columns
    ]

    summary = (
        summary
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "optimizer_summary.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "experiment_results.csv"
    )

    print(
        OUTPUT_DIR
        / "optimizer_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "experiment_histories.csv"
    )

    print(
        "\nFINAL TEST SET WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()