import pandas as pd
import numpy as np
import zipfile
import re
from typing import Optional


BASE_DATA_PATH = "data/processed/df_clean.csv"
ACS_ZIP_PATH = "data/income_pop_edu.zip"

TOTAL_HOUSEHOLDS_PATH = "data/total_households.csv"
CONSTRUCTION_PERMITS_PATH = "data/construction_permits.csv"
RENTAL_VACANCY_PATH = "data/rental_vacancy_rate.csv"

OUTPUT_PATH = "data/processed/df_clean_with_all_features_model_ready_2010_2024.csv"


STATE_TO_DIVISION = {
    # New England
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",

    # Middle Atlantic
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",

    # East North Central
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",

    # West North Central
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",

    # South Atlantic
    "DE": "South Atlantic", "DC": "South Atlantic", "FL": "South Atlantic",
    "GA": "South Atlantic", "MD": "South Atlantic", "NC": "South Atlantic",
    "SC": "South Atlantic", "VA": "South Atlantic", "WV": "South Atlantic",

    # East South Central
    "AL": "East South Central", "KY": "East South Central",
    "MS": "East South Central", "TN": "East South Central",

    # West South Central
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central",
    "TX": "West South Central",

    # Mountain
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",

    # Pacific
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}

def to_numeric(series: pd.Series) -> pd.Series:
    """Convert ACS columns to numeric safely."""
    return pd.to_numeric(series, errors="coerce")

def extract_state_abbr(name: str) -> Optional[str]:
    """
    ACS NAME usually looks like:
    'Jefferson County, Alabama'
    'Pima County, Arizona'
    We use the state name at the end and map to abbreviation.
    """
    if pd.isna(name):
        return None

    state_name = str(name).split(",")[-1].strip()

    state_name_to_abbr = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
        "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
        "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
        "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
        "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    }

    return state_name_to_abbr.get(state_name)

def build_acs_features_from_zip(zip_path: str) -> pd.DataFrame:
    """
    Build division-year ACS feature table from annual county-level ACS S0201 files.
    Features:
      - total_population
      - median_income
      - bachelors_pct
    Handles the ACS naming change starting in 2016, where columns become zero-padded.
    """
    yearly_frames = []

    def pick_col(df: pd.DataFrame, *candidates: str) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"None of these columns were found: {candidates}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        data_files = sorted(
            [name for name in zf.namelist() if name.endswith(".S0201-Data.csv")]
        )

        for file_name in data_files:
            year_match = re.search(r"ACSSPP1Y(\d{4})\.S0201-Data\.csv", file_name)
            if not year_match:
                continue

            year = int(year_match.group(1))

            with zf.open(file_name) as f:
                df = pd.read_csv(f)

            if "GEO_ID" in df.columns:
                df = df[df["GEO_ID"] != "Geography"].copy()

            # 2010-2015 use old names like S0201_006E
            # 2016+ use zero-padded names like S0201_0006E
            pop_col = pick_col(df, "S0201_006E", "S0201_0006E")
            bach_col = pick_col(df, "S0201_099E", "S0201_0099E")
            income_col = pick_col(df, "S0201_214E", "S0201_0214E")

            needed_cols = ["GEO_ID", "NAME", pop_col, bach_col, income_col]
            df = df[needed_cols].copy()

            # Convert numeric columns
            df[pop_col] = to_numeric(df[pop_col])
            df[bach_col] = to_numeric(df[bach_col])
            df[income_col] = to_numeric(df[income_col])

            # Extract state and map to division
            df["state_abbr"] = df["NAME"].apply(extract_state_abbr)
            df["division"] = df["state_abbr"].map(STATE_TO_DIVISION)

            # Keep only rows with valid division and non-missing core values
            df = df.dropna(subset=["division", pop_col]).copy()

            # Rename into project-friendly names
            df = df.rename(columns={
                pop_col: "total_population_county",
                bach_col: "bachelors_pct_county",
                income_col: "median_income_county",
            })

            df["year"] = year

            grouped = (
                df.groupby(["division", "year"], as_index=False)
                  .apply(
                      lambda g: pd.Series({
                          "total_population": g["total_population_county"].sum(),
                          "median_income": np.average(
                              g.loc[g["median_income_county"].notna(), "median_income_county"],
                              weights=g.loc[g["median_income_county"].notna(), "total_population_county"]
                          ) if g["median_income_county"].notna().any() else np.nan,
                          "bachelors_pct": np.average(
                              g.loc[g["bachelors_pct_county"].notna(), "bachelors_pct_county"],
                              weights=g.loc[g["bachelors_pct_county"].notna(), "total_population_county"]
                          ) if g["bachelors_pct_county"].notna().any() else np.nan,
                      })
                  )
                  .reset_index(drop=True)
            )

            yearly_frames.append(grouped)

    acs_features = pd.concat(yearly_frames, ignore_index=True)
    acs_features = acs_features.sort_values(["division", "year"]).reset_index(drop=True)
    return acs_features


def load_feature_csv(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df["division"] = df["division"].astype(str)

    df = df.dropna(subset=["year", "division", value_col]).copy()
    df["year"] = df["year"].astype(int)

    # chop off 2025 and keep only 2010+
    df = df[(df["year"] >= 2010) & (df["year"] <= 2024)].copy()

    return df[["division", "year", value_col]]


def merge_acs_into_panel(base_path: str, acs_zip_path: str, output_path: str) -> pd.DataFrame:
    """Merge ACS features + 3 annual CSV features into main monthly panel."""
    base = pd.read_csv(base_path)

    # Parse dates and create year
    base["date"] = pd.to_datetime(base["date"])
    base["year"] = base["date"].dt.year

    # Optional but recommended: remove duplicate division-date rows
    base = (
        base.groupby(["division", "date", "year"], as_index=False)
        .agg(
            zhvi=("zhvi", "mean"),
            hpi=("hpi", "mean"),
            unemployment_rate=("unemployment_rate", "mean"),
        )
        .sort_values(["division", "date"])
        .reset_index(drop=True)
    )

    # Build ACS division-year features
    acs = build_acs_features_from_zip(acs_zip_path)

    # Load your 3 generated CSVs
    households = load_feature_csv(TOTAL_HOUSEHOLDS_PATH, "total_households")
    permits = load_feature_csv(CONSTRUCTION_PERMITS_PATH, "construction_permits")
    vacancy = load_feature_csv(RENTAL_VACANCY_PATH, "rental_vacancy_rate")

    # Keep only post-2010 and chop off 2025
    base = base[(base["year"] >= 2010) & (base["year"] <= 2024)].copy()
    acs = acs[(acs["year"] >= 2010) & (acs["year"] <= 2024)].copy()

    # Merge onto monthly panel
    merged = (
        base.merge(acs, on=["division", "year"], how="left")
            .merge(households, on=["division", "year"], how="left")
            .merge(permits, on=["division", "year"], how="left")
            .merge(vacancy, on=["division", "year"], how="left")
    )

    # Forward-fill annual variables within each division
    annual_cols = [
        "total_population",
        "median_income",
        "bachelors_pct",
        "total_households",
        "construction_permits",
        "rental_vacancy_rate",
    ]

    merged = merged.sort_values(["division", "date"]).copy()
    merged[annual_cols] = (
        merged.groupby("division")[annual_cols]
              .ffill()
    )

    merged = merged.dropna(subset=annual_cols).copy()

    merged = merged.drop(columns=["year"])

    merged.to_csv(output_path, index=False)

    return merged


if __name__ == "__main__":
    final_df = merge_acs_into_panel(
        base_path=BASE_DATA_PATH,
        acs_zip_path=ACS_ZIP_PATH,
        output_path=OUTPUT_PATH,
    )

    print("Saved:", OUTPUT_PATH)
    print("Shape:", final_df.shape)
    print("Columns:", final_df.columns.tolist())
    print(final_df.head())
