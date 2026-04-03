from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main() -> None:
    here = Path(__file__).resolve().parent
    model_path = here / "outputs" / "model.joblib"
    data_path = here / "ai4i2020.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}. Run train_local.py first.")

    df = pd.read_csv(data_path)
    df["temp_delta_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(model_path)

    # Permutation importance works cleanly even when preprocessing is inside a Pipeline.
    r = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    out_path = here / "outputs" / "permutation_importance.csv"
    importance_df.to_csv(out_path, index=False)
    print(f"Saved permutation importance to: {out_path}")

    engineered_row = importance_df[importance_df["feature"] == "temp_delta_k"]
    if not engineered_row.empty:
        rank = int(engineered_row.index[0]) + 1
        mean = float(engineered_row["importance_mean"].iloc[0])
        print(f"Engineered feature 'temp_delta_k' rank={rank}, importance_mean={mean:.6f}")


if __name__ == "__main__":
    main()

