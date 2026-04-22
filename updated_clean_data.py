import pandas as pd

# Input files
DF_CLEAN_FILE = "data/processed/df_clean.csv"
HOUSEHOLDS_FILE = "data/total_households.csv"
PERMITS_FILE = "data/construction_permits.csv"
VACANCY_FILE = "data/rental_vacancy_rate.csv"

# Output file
OUTPUT_FILE = "data/processed/updated_df_clean.csv"

def main() -> None:
    # -----------------------------
    # 1. Load original monthly file
    # -----------------------------
    df = pd.read_csv(DF_CLEAN_FILE)

    # Parse date and extract year
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # ---------------------------------------------------------
    # 2. Clean duplicate division-date rows in original df_clean
    # ---------------------------------------------------------
    # Earlier sample showed duplicate division/date rows with differing hpi.
    # Aggregate duplicates into one row per division-date.
    df_cleaned = (
        df.groupby(["division", "date", "year"], as_index=False)
        .agg(
            zhvi=("zhvi", "mean"),
            hpi=("hpi", "mean"),
            unemployment_rate=("unemployment_rate", "mean"),
        )
        .sort_values(["division", "date"])
        .reset_index(drop=True)
    )

    # --------------------------------
    # 3. Load annual feature CSV files
    # --------------------------------
    households = pd.read_csv(HOUSEHOLDS_FILE)
    permits = pd.read_csv(PERMITS_FILE)
    vacancy = pd.read_csv(VACANCY_FILE)

    # Standardize column types
    for temp_df in [households, permits, vacancy]:
        temp_df["year"] = pd.to_numeric(temp_df["year"], errors="coerce").astype(int)
        temp_df["division"] = temp_df["division"].astype(str)

    # ------------------------------------------------------
    # 4. Restrict to common overlap across all annual series
    # ------------------------------------------------------
    household_years = set(households["year"].unique())
    permits_years = set(permits["year"].unique())
    vacancy_years = set(vacancy["year"].unique())

    common_years = sorted(household_years & permits_years & vacancy_years)

    # Keep only common years in all datasets
    households = households[households["year"].isin(common_years)].copy()
    permits = permits[permits["year"].isin(common_years)].copy()
    vacancy = vacancy[vacancy["year"].isin(common_years)].copy()
    df_cleaned = df_cleaned[df_cleaned["year"].isin(common_years)].copy()

    # ------------------------------------------
    # 5. Merge annual features into monthly data
    # ------------------------------------------
    updated = df_cleaned.merge(
        households,
        on=["year", "division"],
        how="inner"
    ).merge(
        permits,
        on=["year", "division"],
        how="inner"
    ).merge(
        vacancy,
        on=["year", "division"],
        how="inner"
    )

    # --------------------------------------
    # 6. Reorder columns for cleaner output
    # --------------------------------------
    updated = updated[
        [
            "division",
            "date",
            "year",
            "zhvi",
            "hpi",
            "unemployment_rate",
            "total_households",
            "construction_permits",
            "rental_vacancy_rate",
        ]
    ].sort_values(["division", "date"]).reset_index(drop=True)

    # ----------------
    # 7. Save and log
    # ----------------
    updated.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")
    print(f"Rows: {len(updated)}")
    print(f"Years kept: {min(common_years)}-{max(common_years)}")
    print("Exact years kept:", common_years)
    print("\nPreview:")
    print(updated.head(12))

if __name__ == "__main__":
    main()