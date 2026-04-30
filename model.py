import os
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


BASE_DATA_PATH = "data/processed/df_clean_with_all_features_model_ready_2010_2024.csv"
NUMERIC_FEATURES = [
    "hpi",
    "unemployment_rate",
    "total_population",
    "median_income",
    "bachelors_pct",
    "total_households",
    "construction_permits",
    "rental_vacancy_rate",
]


def load_data(file_path: str = BASE_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")

    df = pd.read_csv(file_path)

    expected_cols = {"division", "date", "zhvi", *NUMERIC_FEATURES}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df = df.drop_duplicates(subset=["division", "date"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["division", "date"]).reset_index(drop=True)
    df = df.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)

    min_date = df["date"].min()
    df["time_index"] = (
        (df["date"].dt.year - min_date.year) * 12
        + (df["date"].dt.month - min_date.month)
    )
    df["month"] = df["date"].dt.month.astype(str)

    df_model = pd.get_dummies(df, columns=["division", "month"], drop_first=True)
    return df, df_model


def time_train_test_split(df_raw: pd.DataFrame, df_model: pd.DataFrame, split_ratio: float = 0.8):
    unique_dates = sorted(df_raw["date"].unique())
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]

    train_mask = df_model["date"] < split_date
    test_mask = df_model["date"] >= split_date

    train_df = df_model.loc[train_mask].copy()
    test_df = df_model.loc[test_mask].copy()

    return train_df, test_df, split_date


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target = "zhvi"
    drop_cols = ["zhvi", "date"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


def train_model(X_train, y_train, model_type: str = "linear", alpha: Optional[float] = 1.0):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0 if alpha is None else alpha)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

    return y_pred, metrics


def save_outputs(test_df_raw: pd.DataFrame, y_test, y_pred, feature_cols, model, model_name: str):
    os.makedirs("outputs", exist_ok=True)

    results = test_df_raw.copy()
    results["actual_zhvi"] = y_test.values
    results["predicted_zhvi"] = y_pred

    division_cols = [c for c in results.columns if c.startswith("division_")]

    def get_division(row):
        for col in division_cols:
            if row[col] == 1:
                return col.replace("division_", "")
        return "East_North_Central"

    results["division_name"] = results.apply(get_division, axis=1)

    final_output = results[["date", "division_name", "actual_zhvi", "predicted_zhvi"]]
    final_output.to_csv(f"outputs/{model_name}_predictions.csv", index=False)

    values = model.coef_ if hasattr(model, "coef_") else model.feature_importances_
    coef_df = (
        pd.DataFrame({"feature": feature_cols, "value": values})
        .sort_values("value", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    coef_df.to_csv(f"outputs/{model_name}_coefficients.csv", index=False)
    return coef_df


def main():
    print("Loading expanded dataset...")
    df = load_data()

    print("Preprocessing data...")
    df_raw, df_model = preprocess_data(df)
    print(f"Original cleaned shape: {df.shape}")
    print(f"Model-ready shape: {df_model.shape}")

    print("Creating time-based train/test split...")
    train_df, test_df, split_date = time_train_test_split(df_raw, df_model, split_ratio=0.8)
    print(f"Train/Test split date: {pd.to_datetime(split_date).date()}")

    print("Building features...")
    X_train, X_test, y_train, y_test, feature_cols = build_features(train_df, test_df)

    model_configs = [
        {"name": "Linear_Regression", "type": "linear", "alpha": None},
        {"name": "Ridge_Regression", "type": "ridge", "alpha": 1.0},
        {"name": "Random_Forest", "type": "rf", "alpha": None},
    ]

    all_results = []

    for config in model_configs:
        print(f"\n--- Training {config['name']} ---")
        model = train_model(X_train, y_train, model_type=config["type"], alpha=config["alpha"])
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = config["name"]
        all_results.append(metrics)
        save_outputs(test_df, y_test, y_pred, feature_cols, model, model_name=config["name"])

    comparison_df = pd.DataFrame(all_results).set_index("Model")
    print("\n--- FINAL MODEL COMPARISON ---")
    print(comparison_df)


if __name__ == "__main__":
    main()
