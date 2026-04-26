import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(file_path="data/processed/df_clean.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")

    df = pd.read_csv(file_path)

    expected_cols = {"division", "date", "zhvi", "hpi", "unemployment_rate"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def preprocess_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # List of all new numerical features
    numeric_features = [
        "hpi", "unemployment_rate", "total_population", "median_income", 
        "bachelors_pct", "total_households", "construction_permits", "rental_vacancy_rate"
    ]
    
    # Handle Lags 
    df = df.sort_values(["division", "date"])
    df["zhvi_lag1"] = df.groupby("division")["zhvi"].shift(1)
    
    # Drop rows with NaN from lags
    df = df.dropna().reset_index(drop=True)
    
    df["month"] = df["date"].dt.month.astype(str)
    df_model = pd.get_dummies(df, columns=["division", "month"], drop_first=True)
    
    return df, df_model


def time_train_test_split(df_raw, df_model, split_ratio=0.8):
    unique_dates = sorted(df_raw["date"].unique())
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]

    train_mask = df_model["date"] < split_date
    test_mask = df_model["date"] >= split_date

    train_df = df_model.loc[train_mask].copy()
    test_df = df_model.loc[test_mask].copy()

    return train_df, test_df, split_date


def build_features(train_df, test_df):
    target = "zhvi"
    drop_cols = ["zhvi", "date"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return y_pred, metrics


def save_outputs(test_df, y_test, y_pred, feature_cols, model):
    os.makedirs("outputs", exist_ok=True)

    results = test_df[["date"]].copy()
    results["actual_zhvi"] = y_test.values
    results["predicted_zhvi"] = y_pred
    results.to_csv("outputs/predictions.csv", index=False)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=abs, ascending=False)

    coef_df.to_csv("outputs/model_coefficients.csv", index=False)

    return coef_df


def main():
    print("Loading cleaned dataset...")
    df = load_data()

    print("Preprocessing data...")
    df_raw, df_model = preprocess_data(df)

    print(f"Original cleaned shape: {df.shape}")
    print(f"Model-ready shape: {df_model.shape}")

    print("Creating time-based train/test split...")
    train_df, test_df, split_date = time_train_test_split(df_raw, df_model, split_ratio=0.8)
    print(f"Train/Test split date: {pd.to_datetime(split_date).date()}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("Building features...")
    X_train, X_test, y_train, y_test, feature_cols = build_features(train_df, test_df)

    print("Training multiple linear regression model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    print("\n--- Model Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSaving outputs...")
    coef_df = save_outputs(test_df, y_test, y_pred, feature_cols, model)

    print("\n--- Top 15 Coefficients by Absolute Magnitude ---")
    print(coef_df.head(15).to_string(index=False))

    print("\nDone.")
    print("Saved files:")
    print("- outputs/predictions.csv")
    print("- outputs/model_coefficients.csv")


if __name__ == "__main__":
    main()