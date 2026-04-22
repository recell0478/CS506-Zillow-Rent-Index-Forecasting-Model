import pandas as pd
import requests

OUTPUT_FILE = "rental_vacancy_rate.csv"

# Standard ACS 1-year years.
# 2020 is excluded because standard ACS 1-year estimates were replaced
# by experimental products.
YEARS = list(range(2005, 2020)) + list(range(2021, 2025))

# ACS 1-year profile variables
# DP04_0005E = Rental vacancy rate
# DP04_0006E = Total housing units
VACANCY_VAR = "DP04_0005E"
HOUSING_UNITS_VAR = "DP04_0006E"

STATE_TO_DIVISION = {
    # New England
    "09": "New England",
    "23": "New England",
    "25": "New England",
    "33": "New England",
    "44": "New England",
    "50": "New England",

    # Middle Atlantic
    "34": "Middle Atlantic",
    "36": "Middle Atlantic",
    "42": "Middle Atlantic",

    # East North Central
    "17": "East North Central",
    "18": "East North Central",
    "26": "East North Central",
    "39": "East North Central",
    "55": "East North Central",

    # West North Central
    "19": "West North Central",
    "20": "West North Central",
    "27": "West North Central",
    "29": "West North Central",
    "31": "West North Central",
    "38": "West North Central",
    "46": "West North Central",

    # South Atlantic
    "10": "South Atlantic",
    "11": "South Atlantic",
    "12": "South Atlantic",
    "13": "South Atlantic",
    "24": "South Atlantic",
    "37": "South Atlantic",
    "45": "South Atlantic",
    "51": "South Atlantic",
    "54": "South Atlantic",

    # East South Central
    "01": "East South Central",
    "21": "East South Central",
    "28": "East South Central",
    "47": "East South Central",

    # West South Central
    "05": "West South Central",
    "22": "West South Central",
    "40": "West South Central",
    "48": "West South Central",

    # Mountain
    "04": "Mountain",
    "08": "Mountain",
    "16": "Mountain",
    "30": "Mountain",
    "32": "Mountain",
    "35": "Mountain",
    "49": "Mountain",
    "56": "Mountain",

    # Pacific
    "02": "Pacific",
    "06": "Pacific",
    "15": "Pacific",
    "41": "Pacific",
    "53": "Pacific",
}

def fetch_state_rental_vacancy(year: int) -> pd.DataFrame:
    """
    Fetch state-level ACS 1-year rental vacancy rate and total housing units
    for one year.

    Returns:
        year, state, NAME, rental_vacancy_rate, total_housing_units
    """
    url = f"https://api.census.gov/data/{year}/acs/acs1/profile"
    params = {
        "get": f"NAME,{VACANCY_VAR},{HOUSING_UNITS_VAR}",
        "for": "state:*",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    df["year"] = year
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["rental_vacancy_rate"] = pd.to_numeric(df[VACANCY_VAR], errors="coerce")
    df["total_housing_units"] = pd.to_numeric(df[HOUSING_UNITS_VAR], errors="coerce")

    return df[["year", "state", "NAME", "rental_vacancy_rate", "total_housing_units"]]

def main() -> None:
    yearly_frames = []

    for year in YEARS:
        print(f"Fetching {year}...")
        df_year = fetch_state_rental_vacancy(year)
        yearly_frames.append(df_year)

    all_states = pd.concat(yearly_frames, ignore_index=True)

    # Map states to Census divisions
    all_states["division"] = all_states["state"].map(STATE_TO_DIVISION)
    all_states = all_states.dropna(subset=["division", "rental_vacancy_rate", "total_housing_units"])

    # Weighted average rental vacancy rate by division-year
    # weight = total housing units
    all_states["weighted_rate"] = (
        all_states["rental_vacancy_rate"] * all_states["total_housing_units"]
    )

    division_df = (
        all_states.groupby(["year", "division"], as_index=False)
        .agg(
            weighted_rate_sum=("weighted_rate", "sum"),
            total_housing_units=("total_housing_units", "sum"),
        )
    )

    division_df["rental_vacancy_rate"] = (
        division_df["weighted_rate_sum"] / division_df["total_housing_units"]
    )

    division_df = (
        division_df[["year", "division", "rental_vacancy_rate"]]
        .sort_values(["year", "division"])
        .reset_index(drop=True)
    )

    division_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved to {OUTPUT_FILE}")
    print("\nPreview:")
    print(division_df.head(12))
    print("\nTail:")
    print(division_df.tail(12))

if __name__ == "__main__":
    main()