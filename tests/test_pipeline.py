import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)


from preprocessing import (
    clean_teleconnect_data,
    encode_target,
    prepare_teleconnect_data,
)

from model import (
    build_ann,
    build_optimizer,
)


@pytest.fixture
def sample_dataframe():
    """
    Small Teleconnect-like dataframe for unit testing.
    """

    rows = 40

    return pd.DataFrame(
        {
            "customerID": [
                f"CUST-{i:03d}"
                for i in range(rows)
            ],

            "gender": (
                ["Male", "Female"]
                * 20
            ),

            "SeniorCitizen": (
                [0, 1] * 20
            ),

            "Partner": (
                ["Yes", "No"] * 20
            ),

            "Dependents": (
                ["No", "Yes"] * 20
            ),

            "tenure": list(
                range(rows)
            ),

            "PhoneService": (
                ["Yes", "No"] * 20
            ),

            "MultipleLines": (
                ["No", "No phone service"]
                * 20
            ),

            "InternetService": (
                ["Fiber optic", "DSL"]
                * 20
            ),

            "OnlineSecurity": (
                ["No", "Yes"] * 20
            ),

            "OnlineBackup": (
                ["Yes", "No"] * 20
            ),

            "DeviceProtection": (
                ["No", "Yes"] * 20
            ),

            "TechSupport": (
                ["No", "Yes"] * 20
            ),

            "StreamingTV": (
                ["Yes", "No"] * 20
            ),

            "StreamingMovies": (
                ["No", "Yes"] * 20
            ),

            "Contract": (
                ["Month-to-month", "One year"]
                * 20
            ),

            "PaperlessBilling": (
                ["Yes", "No"] * 20
            ),

            "PaymentMethod": (
                [
                    "Electronic check",
                    "Mailed check",
                ]
                * 20
            ),

            "MonthlyCharges": np.linspace(
                20,
                100,
                rows,
            ),

            "TotalCharges": [
                " "
                if i == 0
                else str(
                    25.0 * i
                )
                for i in range(rows)
            ],

            "Churn": (
                ["Yes", "No"]
                * 20
            ),
        }
    )


def test_customer_id_is_removed(
    sample_dataframe,
):
    cleaned = clean_teleconnect_data(
        sample_dataframe
    )

    assert (
        "customerID"
        not in cleaned.columns
    )


def test_blank_total_charges_becomes_zero(
    sample_dataframe,
):
    cleaned = clean_teleconnect_data(
        sample_dataframe
    )

    assert (
        cleaned.loc[
            0,
            "TotalCharges",
        ]
        == 0.0
    )


def test_no_missing_values_remain(
    sample_dataframe,
):
    cleaned = clean_teleconnect_data(
        sample_dataframe
    )

    assert (
        cleaned
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_churn_target_encoding(
    sample_dataframe,
):
    cleaned = clean_teleconnect_data(
        sample_dataframe
    )

    _, y = encode_target(
        cleaned
    )

    assert set(
        y.unique()
    ) == {0, 1}

    assert y.iloc[0] == 1
    assert y.iloc[1] == 0


def test_processed_feature_dimensions_are_consistent(
    sample_dataframe,
):
    """
    Encoded feature count depends on which categorical levels
    exist in the supplied dataset. The important requirement is
    that train, validation, and test use the same fitted encoder.
    """
    prepared = prepare_teleconnect_data(
        sample_dataframe,
        random_state=42,
    )

    feature_count = (
        prepared.X_train.shape[1]
    )

    assert feature_count > 0

    assert (
        prepared.X_val.shape[1]
        == feature_count
    )

    assert (
        prepared.X_test.shape[1]
        == feature_count
    )

    assert (
        len(prepared.feature_names)
        == feature_count
    )


def test_numeric_training_features_are_scaled(
    sample_dataframe,
):
    prepared = prepare_teleconnect_data(
        sample_dataframe,
        random_state=42,
    )

    numeric_train = (
        prepared.X_train[
            :,
            :4,
        ]
    )

    assert np.allclose(
        numeric_train.mean(
            axis=0
        ),
        0,
        atol=1e-6,
    )

    assert np.allclose(
        numeric_train.std(
            axis=0
        ),
        1,
        atol=1e-6,
    )


def test_stratification_preserves_churn_rate(
    sample_dataframe,
):
    prepared = prepare_teleconnect_data(
        sample_dataframe,
        random_state=42,
    )

    overall_rate = 0.50

    assert abs(
        prepared.y_train.mean()
        - overall_rate
    ) < 0.10

    assert abs(
        prepared.y_val.mean()
        - overall_rate
    ) < 0.10

    assert abs(
        prepared.y_test.mean()
        - overall_rate
    ) < 0.10


@pytest.mark.parametrize(
    "optimizer_name",
    [
        "adam",
        "rmsprop",
        "sgd",
    ],
)
def test_supported_optimizers_build(
    optimizer_name,
):
    optimizer = build_optimizer(
        optimizer_name
    )

    assert optimizer is not None


def test_ann_outputs_valid_probability():
    model = build_ann(
        input_features=30,
        optimizer_name="adam",
        dropout_rate=0.30,
    )

    sample = np.zeros(
        (1, 30),
        dtype=np.float32,
    )

    prediction = model.predict(
        sample,
        verbose=0,
    )

    assert prediction.shape == (
        1,
        1,
    )

    probability = float(
        prediction[0, 0]
    )

    assert (
        0.0
        <= probability
        <= 1.0
    )