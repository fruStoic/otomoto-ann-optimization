"""
Analyze the completed five-seed optimizer experiment.

This script reads saved validation results only.
It does NOT train models and does NOT access the test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "experiment_results.csv"
)

HISTORIES_PATH = (
    OUTPUT_DIR
    / "experiment_histories.csv"
)


OPTIMIZER_ORDER = [
    "adam",
    "rmsprop",
    "sgd",
]


def main():

    results = pd.read_csv(
        RESULTS_PATH
    )

    histories = pd.read_csv(
        HISTORIES_PATH
    )

    # ---------------------------------------------------------
    # Detailed mean ± SD summary
    # ---------------------------------------------------------

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive",
        "false_negative",
        "best_val_loss",
        "epochs_run",
    ]

    summary = (
        results
        .groupby("optimizer")[
            metrics
        ]
        .agg(["mean", "std"])
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
        / "optimizer_summary_detailed.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Apply the predefined selection rule:
    #
    # 1. highest mean F1
    # 2. highest mean recall
    # 3. highest mean precision
    # ---------------------------------------------------------

    ranking = (
        summary
        .sort_values(
            by=[
                "f1_mean",
                "recall_mean",
                "precision_mean",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    selected_optimizer = (
        ranking.loc[
            0,
            "optimizer",
        ]
    )

    print(
        "\nDETAILED OPTIMIZER SUMMARY"
    )

    print(
        "=" * 80
    )

    for optimizer in OPTIMIZER_ORDER:

        row = summary[
            summary["optimizer"]
            == optimizer
        ].iloc[0]

        print(
            f"\n{optimizer.upper()}"
        )

        print(
            f"Accuracy : "
            f"{row['accuracy_mean']:.3f} "
            f"± {row['accuracy_std']:.3f}"
        )

        print(
            f"Precision: "
            f"{row['precision_mean']:.3f} "
            f"± {row['precision_std']:.3f}"
        )

        print(
            f"Recall   : "
            f"{row['recall_mean']:.3f} "
            f"± {row['recall_std']:.3f}"
        )

        print(
            f"F1       : "
            f"{row['f1_mean']:.3f} "
            f"± {row['f1_std']:.3f}"
        )

        print(
            f"ROC-AUC  : "
            f"{row['roc_auc_mean']:.3f} "
            f"± {row['roc_auc_std']:.3f}"
        )

        print(
            f"FP       : "
            f"{row['false_positive_mean']:.1f} "
            f"± "
            f"{row['false_positive_std']:.1f}"
        )

        print(
            f"FN       : "
            f"{row['false_negative_mean']:.1f} "
            f"± "
            f"{row['false_negative_std']:.1f}"
        )

        print(
            f"Epochs   : "
            f"{row['epochs_run_mean']:.1f} "
            f"± "
            f"{row['epochs_run_std']:.1f}"
        )

    print(
        "\nSELECTED OPTIMIZER"
    )

    print(
        "=" * 80
    )

    print(
        selected_optimizer.upper()
    )

    # ---------------------------------------------------------
    # Quantify SGD vs Adam difference.
    # ---------------------------------------------------------

    adam = summary[
        summary["optimizer"]
        == "adam"
    ].iloc[0]

    sgd = summary[
        summary["optimizer"]
        == "sgd"
    ].iloc[0]

    f1_difference = (
        sgd["f1_mean"]
        - adam["f1_mean"]
    )

    print(
        "\nSGD vs Adam mean F1 difference:"
    )

    print(
        f"{f1_difference:.4f}"
    )

    print(
        "This difference should be interpreted "
        "against the observed run-to-run "
        "standard deviations."
    )

    # ---------------------------------------------------------
    # Figure 1: F1 mean ± SD
    # ---------------------------------------------------------

    plot_summary = (
        summary
        .set_index("optimizer")
        .loc[OPTIMIZER_ORDER]
        .reset_index()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        plot_summary["optimizer"],
        plot_summary["f1_mean"],
        yerr=plot_summary["f1_std"],
        capsize=6,
    )

    plt.ylabel(
        "Validation F1-score"
    )

    plt.xlabel(
        "Optimizer"
    )

    plt.title(
        "Optimizer Validation F1 "
        "(Mean ± SD Across Five Seeds)"
    )

    plt.ylim(
        0.50,
        0.66,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "optimizer_f1_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 2: convergence cost
    # ---------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        plot_summary["optimizer"],
        plot_summary[
            "epochs_run_mean"
        ],
        yerr=plot_summary[
            "epochs_run_std"
        ],
        capsize=6,
    )

    plt.ylabel(
        "Epochs until training stopped"
    )

    plt.xlabel(
        "Optimizer"
    )

    plt.title(
        "Optimizer Convergence "
        "(Mean ± SD Across Five Seeds)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "optimizer_epochs_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 3:
    # representative validation-loss curves.
    #
    # Seed 42 was specified before results were observed,
    # so it is not selected because it happened to look best.
    # ---------------------------------------------------------

    seed_42 = histories[
        histories["seed"] == 42
    ]

    plt.figure(
        figsize=(8, 5)
    )

    for optimizer in OPTIMIZER_ORDER:

        run = seed_42[
            seed_42["optimizer"]
            == optimizer
        ]

        plt.plot(
            run["epoch"],
            run["val_loss"],
            label=optimizer.upper(),
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation binary cross-entropy"
    )

    plt.title(
        "Validation Loss by Optimizer "
        "(Seed 42)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "optimizer_loss_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Record the selection decision.
    # ---------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "optimizer_selection.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "SELECTED OPTIMIZER\n"
        )

        file.write(
            "=" * 50
            + "\n"
        )

        file.write(
            f"{selected_optimizer.upper()}\n\n"
        )

        file.write(
            "Selection rule:\n"
        )

        file.write(
            "1. Highest mean validation F1\n"
        )

        file.write(
            "2. Highest mean validation recall\n"
        )

        file.write(
            "3. Highest mean validation precision\n\n"
        )

        file.write(
            f"SGD - Adam mean F1 difference: "
            f"{f1_difference:.4f}\n"
        )

        file.write(
            "\nInterpretation:\n"
        )

        file.write(
            "SGD is the nominal winner under the "
            "predefined selection rule, but its "
            "advantage over Adam is small relative "
            "to the observed stochastic variation. "
            "Adam converges substantially faster."
        )

    print(
        "\nFigures and detailed summary saved."
    )

    print(
        "\nTEST SET WAS NOT USED."
    )


if __name__ == "__main__":
    main()