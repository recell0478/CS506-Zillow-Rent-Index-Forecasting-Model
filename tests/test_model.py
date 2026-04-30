import unittest
from pathlib import Path

import pandas as pd

from model import (
    build_features,
    evaluate_model,
    load_data,
    preprocess_data,
    time_train_test_split,
    train_model,
)


def make_sample_dataframe():
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    rows = []
    for division, base in [("Pacific", 100.0), ("New England", 200.0)]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "division": division,
                    "date": date.strftime("%Y-%m-%d"),
                    "zhvi": base + idx * 10,
                    "hpi": 50 + idx,
                    "unemployment_rate": 4.0 + idx * 0.1,
                    "total_population": 1000000 + idx,
                    "median_income": 60000 + idx,
                    "bachelors_pct": 35.0 + idx * 0.1,
                    "total_households": 400000 + idx,
                    "construction_permits": 10000 + idx,
                    "rental_vacancy_rate": 6.0 + idx * 0.1,
                }
            )
    return pd.DataFrame(rows)


class ModelPipelineTests(unittest.TestCase):
    def test_load_data_reads_expected_repo_dataset(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "data" / "processed" / "df_clean_with_all_features_model_ready_2010_2024.csv"

        df = load_data(str(dataset_path))

        self.assertFalse(df.empty)
        self.assertTrue(
            {
                "division",
                "date",
                "zhvi",
                "hpi",
                "unemployment_rate",
                "total_population",
                "median_income",
                "bachelors_pct",
                "total_households",
                "construction_permits",
                "rental_vacancy_rate",
            }.issubset(df.columns)
        )

    def test_preprocess_data_creates_model_ready_features(self):
        df = make_sample_dataframe()
        df_raw, df_model = preprocess_data(df)

        self.assertIn("time_index", df_raw.columns)
        self.assertTrue(any(col.startswith("division_") for col in df_model.columns))
        self.assertTrue(any(col.startswith("month_") for col in df_model.columns))
        self.assertFalse(df_model.isna().any().any())

    def test_time_split_respects_chronology(self):
        df = make_sample_dataframe()
        df_raw, df_model = preprocess_data(df)
        train_df, test_df, split_date = time_train_test_split(df_raw, df_model, split_ratio=0.5)

        self.assertFalse(train_df.empty)
        self.assertFalse(test_df.empty)
        self.assertLess(train_df["date"].max(), split_date)
        self.assertGreaterEqual(test_df["date"].min(), split_date)

    def test_training_pipeline_runs_end_to_end(self):
        df = make_sample_dataframe()
        df_raw, df_model = preprocess_data(df)
        train_df, test_df, _ = time_train_test_split(df_raw, df_model, split_ratio=0.5)
        X_train, X_test, y_train, y_test, _ = build_features(train_df, test_df)

        model = train_model(X_train, y_train, model_type="ridge", alpha=1.0)
        predictions, metrics = evaluate_model(model, X_test, y_test)

        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(set(metrics.keys()), {"MAE", "RMSE", "R2"})
        self.assertTrue(all(pd.notna(list(metrics.values()))))


if __name__ == "__main__":
    unittest.main()
