"""
Training script for the medical insurance cost prediction model.

This script rebuilds the saved scikit-learn pipeline from the processed dataset
and writes the trained model to `model.pkl`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "processed_insurance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
TARGET_COLUMN = "charges"
OPTIONAL_DROP_COLUMNS = ["Weight_Condition"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    categorical_columns = ["sex", "smoker", "region"]
    scaled_numeric_columns = ["bmi"]

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_columns,
            ),
            ("num", StandardScaler(), scaled_numeric_columns),
        ],
        remainder="passthrough",
    )


def build_pipeline() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        max_features="log2",
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def rmse_score(actual: pd.Series, predicted) -> float:
    mse = mean_squared_error(actual, predicted)
    return float(mse) ** 0.5


def train(save_path: Path = MODEL_PATH) -> Pipeline:
    print("Loading data...")
    dataset = load_data()

    features = dataset.drop(columns=[TARGET_COLUMN, *OPTIONAL_DROP_COLUMNS], errors="ignore")
    target = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=45,
    )

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    print(f"\n{'=' * 45}")
    print("TRAINING SET")
    print(f"RMSE : {rmse_score(y_train, train_predictions):,.2f}")
    print(f"R2   : {r2_score(y_train, train_predictions):.4f}")
    print("\nTEST SET")
    print(f"RMSE : {rmse_score(y_test, test_predictions):,.2f}")
    print(f"R2   : {r2_score(y_test, test_predictions):.4f}")
    print(f"{'=' * 45}\n")

    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")
    return pipeline


if __name__ == "__main__":
    train()
