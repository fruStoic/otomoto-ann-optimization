"""
ANN model definitions for the Teleconnect churn optimization experiment.

The architecture is held constant across optimizer experiments so that
differences in validation performance can be attributed to optimizer choice
rather than changes in network capacity.
"""

import tensorflow as tf


OPTIMIZER_CONFIGS = {
    "adam": {
        "learning_rate": 0.001,
    },
    "rmsprop": {
        "learning_rate": 0.001,
    },
    "sgd": {
        "learning_rate": 0.01,
    },
}


def build_optimizer(
    optimizer_name: str,
) -> tf.keras.optimizers.Optimizer:
    """
    Construct one of the three optimizers being compared.

    Learning rates are explicitly declared rather than relying on
    implicit library defaults.
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name not in OPTIMIZER_CONFIGS:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    learning_rate = (
        OPTIMIZER_CONFIGS[
            optimizer_name
        ]["learning_rate"]
    )

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )


def build_ann(
    input_features: int,
    optimizer_name: str,
    dropout_rate: float = 0.30,
) -> tf.keras.Model:
    """
    Build the fixed ANN architecture used for every optimizer.

    Architecture:
        Input
        -> Dense(32, ReLU)
        -> Dropout(0.30)
        -> Dense(16, ReLU)
        -> Dense(1, Sigmoid)
    """

    if input_features <= 0:
        raise ValueError(
            "input_features must be greater than zero."
        )

    if not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must be between 0 and 1."
        )

    optimizer = build_optimizer(
        optimizer_name
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_features,),
                name="input_features",
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
                name="hidden_32",
            ),

            tf.keras.layers.Dropout(
                dropout_rate,
                name="dropout",
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
                name="hidden_16",
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name="churn_probability",
            ),
        ],
        name=(
            f"teleconnect_{optimizer_name}_ann"
        ),
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
        ],
    )

    return model


def create_early_stopping(
    patience: int = 10,
) -> tf.keras.callbacks.EarlyStopping:
    """
    Stop when validation loss ceases improving and restore
    the weights from the best validation-loss epoch.
    """

    if patience < 1:
        raise ValueError(
            "patience must be at least 1."
        )

    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=0,
    )