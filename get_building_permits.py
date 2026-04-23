import io
import pandas as pd
import requests

SOURCE_URL = "https://www.census.gov/construction/bps/xls/annualhistorybystate.xls"
OUTPUT_FILE = "construction_permits.csv"

START_YEAR = 2010
END_YEAR = 2024

# Map worksheet abbreviation directly to Census division
SHEET_TO_DIVISION = {
    # New England
    "CT": "New England",
    "ME": "New England",
    "MA": "New England",
    "NH": "New England",
    "RI": "New England",
    "VT": "New England",

    # Middle Atlantic
    "NJ": "Middle Atlantic",
    "NY": "Middle Atlantic",
    "PA": "Middle Atlantic",

    # East North Central
    "IL": "East North Central",
    "IN": "East North Central",
    "MI": "East North Central",
    "OH": "East North Central",
    "WI": "East North Central",

    # West North Central
    "IA": "West North Central",
    "KS": "West North Central",
    "MN": "West North Central",
    "MO": "West North Central",
    "NE": "West North Central",
    "ND": "West North Central",
    "SD": "West North Central",

    # South Atlantic
    "DE": "South Atlantic",
    "DC": "South Atlantic",
    "FL": "South Atlantic",
    "GA": "South Atlantic",
    "MD": "South Atlantic",
    "NC": "South Atlantic",
    "SC": "South Atlantic",
    "VA": "South Atlantic",
    "WV": "South Atlantic",

    # East South Central
    "AL": "East South Central",
    "KY": "East South Central",
    "MS": "East South Central",
    "TN": "East South Central",

    # West South Central
    "AR": "West South Central",
    "LA": "West South Central",
    "OK": "West South Central",
    "TX": "West South Central",

    # Mountain
    "AZ": "Mountain",
    "CO": "Mountain",
    "ID": "Mountain",
    "MT": "Mountain",
    "NV": "Mountain",
    "NM": "Mountain",
    "UT": "Mountain",
    "WY": "Mountain",

    # Pacific
    "AK": "Pacific",
    "CA": "Pacific",
    "HI": "Pacific",
    "OR": "Pacific",
    "WA": "Pacific",
}

def extract_state_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    For each state sheet:
    - column 1 = actual year
    - column 3 = Total Units -> Housing Units
    """
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    year_col = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    permit_col = pd.to_numeric(df.iloc[:, 3], errors="coerce")

    mask = (
        year_col.between(START_YEAR, END_YEAR, inclusive="both")
        & permit_col.notna()
    )

    state_rows = pd.DataFrame({
        "year": year_col[mask].astype(int),
        "construction_permits": permit_col[mask].astype(float),
        "division": SHEET_TO_DIVISION[sheet_name],
    })

    return state_rows

def main() -> None:
    print("Downloading source workbook...")
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()

    print("Opening workbook...")
    xls = pd.ExcelFile(io.BytesIO(response.content), engine="xlrd")
    print("Sheets found:", xls.sheet_names)

    state_frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name in SHEET_TO_DIVISION:
            print(f"Reading {sheet_name}...")
            state_df = extract_state_sheet(xls, sheet_name)
            state_frames.append(state_df)

    if not state_frames:
        raise ValueError("No usable state sheets were found in the workbook.")

    all_states_df = pd.concat(state_frames, ignore_index=True)

    # Optional safety cleanup
    all_states_df = all_states_df.drop_duplicates()

    division_df = (
        all_states_df.groupby(["year", "division"], as_index=False)["construction_permits"]
        .sum()
        .sort_values(["year", "division"])
        .reset_index(drop=True)
    )

    division_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to {OUTPUT_FILE}")
    print("\nPreview:")
    print(division_df.tail(20))

if __name__ == "__main__":
    main()