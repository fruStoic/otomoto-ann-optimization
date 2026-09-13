from data_loader import load_teleconnect_data

from preprocessing import (
    clean_teleconnect_data,
    encode_target,
    prepare_teleconnect_data,
)


def class_summary(y):
    total = len(y)
    churn = int(y.sum())
    retained = total - churn

    return {
        "total": total,
        "no_churn": retained,
        "churn": churn,
        "churn_percent": (
            churn / total * 100
        ),
    }


def main():

    df = load_teleconnect_data()

    print("\nRAW DATA")
    print("-" * 60)

    print(
        f"Shape: {df.shape}"
    )

    blank_total_charges = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Blank TotalCharges values: "
        f"{blank_total_charges}"
    )

    cleaned = clean_teleconnect_data(
        df
    )

    X, y = encode_target(
        cleaned
    )

    print("\nCLEANED DATA")
    print("-" * 60)

    print(
        f"Shape: {cleaned.shape}"
    )

    print(
        f"Missing values: "
        f"{cleaned.isna().sum().sum()}"
    )

    print(
        f"Predictor columns: "
        f"{X.shape[1]}"
    )

    print(
        f"TotalCharges minimum: "
        f"{cleaned['TotalCharges'].min():.2f}"
    )

    prepared = prepare_teleconnect_data(
        df,
        random_state=42,
    )

    print("\nPROCESSED SPLITS")
    print("-" * 60)

    print(
        f"Encoded feature count: "
        f"{len(prepared.feature_names)}"
    )

    print(
        "Train:",
        prepared.X_train.shape,
        class_summary(
            prepared.y_train
        ),
    )

    print(
        "Validation:",
        prepared.X_val.shape,
        class_summary(
            prepared.y_val
        ),
    )

    print(
        "Test:",
        prepared.X_test.shape,
        class_summary(
            prepared.y_test
        ),
    )

    print("\nSCALING CHECK")
    print("-" * 60)

    # The first four transformed columns are numeric.
    print(
        "Numeric training means:",
        prepared.X_train[
            :, :4
        ].mean(axis=0),
    )

    print(
        "Numeric training std:",
        prepared.X_train[
            :, :4
        ].std(axis=0),
    )

    print("\nFIRST 15 ENCODED FEATURES")
    print("-" * 60)

    for feature in (
        prepared.feature_names[:15]
    ):
        print(feature)


if __name__ == "__main__":
    main()