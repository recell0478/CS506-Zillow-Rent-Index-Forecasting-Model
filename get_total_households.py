import pandas as pd
import requests

OUTPUT_FILE = "total_households.csv"

# ACS 5-year years
YEARS = list(range(2010, 2025))

# ACS 5-year profile variable:
# DP02_0001E = Estimate!!HOUSEHOLDS BY TYPE!!Total households
VARIABLE = "DP02_0001E"

STATE_TO_DIVISION = {
    "09": "New England", "23": "New England", "25": "New England",
    "33": "New England", "44": "New England", "50": "New England",

    "34": "Middle Atlantic", "36": "Middle Atlantic", "42": "Middle Atlantic",

    "17": "East North Central", "18": "East North Central",
    "26": "East North Central", "39": "East North Central",
    "55": "East North Central",

    "19": "West North Central", "20": "West North Central",
    "27": "West North Central", "29": "West North Central",
    "31": "West North Central", "38": "West North Central",
    "46": "West North Central",

    "10": "South Atlantic", "11": "South Atlantic", "12": "South Atlantic",
    "13": "South Atlantic", "24": "South Atlantic", "37": "South Atlantic",
    "45": "South Atlantic", "51": "South Atlantic", "54": "South Atlantic",

    "01": "East South Central", "21": "East South Central",
    "28": "East South Central", "47": "East South Central",

    "05": "West South Central", "22": "West South Central",
    "40": "West South Central", "48": "West South Central",

    "04": "Mountain", "08": "Mountain", "16": "Mountain",
    "30": "Mountain", "32": "Mountain", "35": "Mountain",
    "49": "Mountain", "56": "Mountain",

    "02": "Pacific", "06": "Pacific", "15": "Pacific",
    "41": "Pacific", "53": "Pacific",
}

def fetch_state_households(year: int) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
    params = {
        "get": f"NAME,{VARIABLE}",
        "for": "state:*",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    df["year"] = year
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["total_households"] = pd.to_numeric(df[VARIABLE], errors="coerce")

    return df[["year", "state", "NAME", "total_households"]]

def main() -> None:
    yearly_frames = []

    for year in YEARS:
        print(f"Fetching {year}...")
        df_year = fetch_state_households(year)
        yearly_frames.append(df_year)

    all_states = pd.concat(yearly_frames, ignore_index=True)
    all_states["division"] = all_states["state"].map(STATE_TO_DIVISION)
    all_states = all_states.dropna(subset=["division", "total_households"])

    division_df = (
        all_states.groupby(["year", "division"], as_index=False)["total_households"]
        .sum()
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