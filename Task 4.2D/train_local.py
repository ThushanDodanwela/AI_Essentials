import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create at least one derived feature.

    temp_delta_k = Process temperature - Air temperature
    Physically: when the process runs hotter relative to ambient air, the machine is under
    higher thermal stress; sustained thermal stress is linked to wear, lubrication breakdown,
    and higher chance of failure.
    """
    df = df.copy()
    df["temp_delta_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


def main() -> None:
    here = Path(__file__).resolve().parent
    data_path = here / "ai4i2020.csv"
    out_dir = here / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    # Keep the main failure label (binary target). Drop the failure-subtype columns to
    # prevent "label leakage" (they are derived from the same failure event).
    df = add_engineered_features(df)
    y = df["Machine failure"].astype(int)
    X = df.drop(
        columns=[
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
            "UDI",
            "Product ID",
        ]
    )

    categorical_cols = ["Type"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    # Random Forest is robust and simple for beginners.
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")

    # Create a "healthy" and "critical-ish" example payload for later API testing.
    # (You will still validate the predictions after deployment.)
    healthy = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20,
    }
    critical = {
        "Type": "H",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 313.0,  # higher thermal stress (bigger delta)
        "Rotational speed [rpm]": 2500,
        "Torque [Nm]": 80.0,
        "Tool wear [min]": 220,
    }

    sample_path = out_dir / "sample_payloads.json"
    sample_path.write_text(json.dumps({"healthy": healthy, "critical": critical}, indent=2))
    print(f"Wrote sample payloads to: {sample_path}")


if __name__ == "__main__":
    main()

