from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import load_teleconnect_data
from preprocessing import prepare_teleconnect_data
from model import build_ann, create_early_stopping
from evaluation import predict_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]

THRESHOLDS = np.arange(
    0.20,
    0.61,
    0.05,
)

MAX_EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_teleconnect_data()

    prepared = prepare_teleconnect_data(
        df,
        random_state=42,
    )

    rows = []

    for seed in SEEDS:

        tf.keras.backend.clear_session()
        set_seed(seed)

        model = build_ann(
            input_features=(
                prepared.X_train.shape[1]
            ),
            optimizer_name="sgd",
            dropout_rate=0.30,
        )

        model.fit(
            prepared.X_train,
            prepared.y_train,
            validation_data=(
                prepared.X_val,
                prepared.y_val,
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                create_early_stopping(
                    patience=PATIENCE
                )
            ],
            verbose=0,
        )

        probabilities = (
            predict_probabilities(
                model,
                prepared.X_val,
            )
        )

        for threshold in THRESHOLDS:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            rows.append(
                {
                    "seed": seed,
                    "threshold": round(
                        float(threshold),
                        2,
                    ),
                    "precision": (
                        precision_score(
                            prepared.y_val,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    "recall": (
                        recall_score(
                            prepared.y_val,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    "f1": (
                        f1_score(
                            prepared.y_val,
                            predictions,
                            zero_division=0,
                        )
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_DIR
        / "threshold_results.csv",
        index=False,
    )

    summary = (
        results
        .groupby("threshold")[
            [
                "precision",
                "recall",
                "f1",
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

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
        / "threshold_summary.csv",
        index=False,
    )

    print(
        "\nVALIDATION THRESHOLD SWEEP"
    )

    print(
        "=" * 75
    )

    for _, row in summary.iterrows():

        print(
            f"Threshold "
            f"{row['threshold']:.2f} | "
            f"Precision "
            f"{row['precision_mean']:.3f} "
            f"± {row['precision_std']:.3f} | "
            f"Recall "
            f"{row['recall_mean']:.3f} "
            f"± {row['recall_std']:.3f} | "
            f"F1 "
            f"{row['f1_mean']:.3f} "
            f"± {row['f1_std']:.3f}"
        )

    print(
        "\nTEST SET WAS NOT USED."
    )


if __name__ == "__main__":
    main()