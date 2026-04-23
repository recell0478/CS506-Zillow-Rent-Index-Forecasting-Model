import pandas as pd
import numpy as np
import zipfile
import re


BASE_DATA_PATH = "data/processed/df_clean.csv"
ACS_ZIP_PATH = "productDownload_2026-04-18T140321.zip"
OUTPUT_PATH = "data/processed/df_clean_with_acs_model_ready_2010plus.csv"


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

def extract_state_abbr(name: str) -> str | None:
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
    """
    yearly_frames = []

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

            # Keep only required columns
            needed_cols = [
                "GEO_ID",
                "NAME",
                "S0201_006E",   # total population
                "S0201_099E",   # bachelor's degree or higher (%)
                "S0201_214E",   # median household income
            ]
            df = df[needed_cols].copy()

            # Convert numeric columns
            df["S0201_006E"] = to_numeric(df["S0201_006E"])
            df["S0201_099E"] = to_numeric(df["S0201_099E"])
            df["S0201_214E"] = to_numeric(df["S0201_214E"])

            # Extract state and map to division
            df["state_abbr"] = df["NAME"].apply(extract_state_abbr)
            df["division"] = df["state_abbr"].map(STATE_TO_DIVISION)

            # Keep only rows with valid division and non-missing core values
            df = df.dropna(subset=["division", "S0201_006E"]).copy()

            # Rename into project-friendly names
            df = df.rename(columns={
                "S0201_006E": "total_population_county",
                "S0201_099E": "bachelors_pct_county",
                "S0201_214E": "median_income_county",
            })

            df["year"] = year

            # Aggregate county -> division
            # total_population = sum
            # median_income = population-weighted average
            # bachelors_pct = population-weighted average
            grouped = (
                df.groupby(["division", "year"], as_index=False)
                  .apply(
                      lambda g: pd.Series({
                          "total_population": g["total_population_county"].sum(),
                          "median_income": np.average(
                              g["median_income_county"].dropna(),
                              weights=g.loc[g["median_income_county"].notna(), "total_population_county"]
                          ) if g["median_income_county"].notna().any() else np.nan,
                          "bachelors_pct": np.average(
                              g["bachelors_pct_county"].dropna(),
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

def merge_acs_into_panel(base_path: str, acs_zip_path: str, output_path: str) -> pd.DataFrame:
    """Merge ACS features into main monthly panel and create model-ready dataset."""
    base = pd.read_csv(base_path)

    # Parse dates and create year
    base["date"] = pd.to_datetime(base["date"])
    base["year"] = base["date"].dt.year

    # Build ACS division-year features
    acs = build_acs_features_from_zip(acs_zip_path)

    # Merge onto monthly panel
    merged = base.merge(acs, on=["division", "year"], how="left")

    # Keep only post-2010 period because ACS starts in 2010
    merged = merged[merged["year"] >= 2010].copy()

    # Forward-fill ACS variables within each division
    acs_cols = ["total_population", "median_income", "bachelors_pct"]
    merged = merged.sort_values(["division", "date"]).copy()
    merged[acs_cols] = (
        merged.groupby("division")[acs_cols]
              .ffill()
    )

    merged = merged.dropna(subset=acs_cols).copy()

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