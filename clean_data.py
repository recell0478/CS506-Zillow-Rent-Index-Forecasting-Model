import pandas as pd
import os

# -------------------------------
# 1. State → Census Division map
# -------------------------------

state_to_division = {
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central",
    "ND": "West North Central", "SD": "West North Central",
    "DE": "South Atlantic", "FL": "South Atlantic", "GA": "South Atlantic",
    "MD": "South Atlantic", "NC": "South Atlantic", "SC": "South Atlantic",
    "VA": "South Atlantic", "DC": "South Atlantic", "WV": "South Atlantic",
    "AL": "East South Central", "KY": "East South Central",
    "MS": "East South Central", "TN": "East South Central",
    "AR": "West South Central", "LA": "West South Central",
    "OK": "West South Central", "TX": "West South Central",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain",
    "MT": "Mountain", "NV": "Mountain", "NM": "Mountain",
    "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific",
    "OR": "Pacific", "WA": "Pacific"
}

# ---------------------------------------------------------
# 2. ZHVI: Load and Normalize Date (Zillow)
# ---------------------------------------------------------
print("Processing ZHVI (Zillow)...")

zhvi = pd.read_csv("data/zhvi.csv")
zhvi.columns = zhvi.columns.str.strip().str.lower().str.replace(" ", "_")

# Reshape (wide → long)
date_cols = [col for col in zhvi.columns if col[:4].isdigit()]
id_cols = [col for col in zhvi.columns if col not in date_cols]

zhvi_long = zhvi.melt(
    id_vars=id_cols,
    value_vars=date_cols,
    var_name="date",
    value_name="zhvi"
)

# Standardize date to 1st of the month
zhvi_long["date"] = pd.to_datetime(zhvi_long["date"]).dt.to_period('M').dt.to_timestamp()

# National USA stats
us_zhvi = zhvi_long[zhvi_long["regionname"] == "United States"].copy()
us_zhvi["division"] = "USA"
us_zhvi = us_zhvi[["division", "date", "zhvi"]]

# Division stats
zhvi_long = zhvi_long[zhvi_long["regiontype"] == "msa"].dropna(subset=["statename"])
zhvi_long["division"] = zhvi_long["statename"].map(state_to_division)
zhvi_division = zhvi_long.groupby(["division", "date"])["zhvi"].mean().reset_index()

final_zhvi = pd.concat([zhvi_division, us_zhvi], ignore_index=True)

# ---------------------------------------------------------
# 3. HPI: Load and Normalize Date
# ---------------------------------------------------------
print("Processing HPI (FHFA)...")
hpi = pd.read_excel("data/hpi.xlsx", skiprows=3)
hpi.rename(columns={'Month': 'date'}, inplace=True)
hpi['date'] = pd.to_datetime(hpi['date']).dt.to_period('M').dt.to_timestamp()

hpi_long = hpi.melt(id_vars=['date'], var_name='division_raw', value_name='hpi')
hpi_long['division'] = hpi_long['division_raw'].str.split('\n').str[0].str.strip()
hpi_long.loc[hpi_long['division'].str.contains('USA', na=False), 'division'] = 'USA'

final_hpi = hpi_long[['date', 'division', 'hpi']].dropna()

# ---------------------------------------------------------
# 4. UPI: Load and Map to Divisions 
# ---------------------------------------------------------
print("Processing UPI (Unemployment)...")
upi = pd.read_excel("data/upi.xlsx", skiprows=2)
upi.columns = [c.strip() for c in upi.columns] # Clean column headers

# Extract State from "County Name/State Abbreviation" 
upi['state'] = upi['County Name/State Abbreviation'].str.split(',').str[-1].str.strip()
upi['division'] = upi['state'].map(state_to_division)

# Remove 'p' from preliminary data strings like "Dec-24 p"
upi['date_str'] = upi['Period'].str.replace(' p', '')
upi['date'] = pd.to_datetime(upi['date_str'], format='%b-%y')

# Clean unemployment rate column (remove hyphens/errors)
upi_rate_col = 'Unemployment Rate (%)' # If hyphenated in file: 'Unemploy-ment Rate (%)'
if upi_rate_col not in upi.columns:
    upi_rate_col = [c for c in upi.columns if 'Unemploy' in c][0]

upi[upi_rate_col] = pd.to_numeric(upi[upi_rate_col], errors='coerce')

# Aggregate by Division and Date
final_upi = upi.groupby(['division', 'date'])[upi_rate_col].mean().reset_index()
final_upi.rename(columns={upi_rate_col: 'unemployment_rate'}, inplace=True)

# ---------------------------------------------------------
# 5. MERGE: Create the Final Master Dataset
# ---------------------------------------------------------
print("Merging all features...")
# Join ZHVI and HPI
master_df = pd.merge(final_zhvi, final_hpi, on=['date', 'division'], how='inner')
# Join with Unemployment
master_df = pd.merge(master_df, final_upi, on=['date', 'division'], how='left')

master_df = master_df.sort_values(by=['division', 'date'])
master_df.to_csv("data/processed/df_clean.csv", index=False)

print(f"Success! Final Dataset with columns: {master_df.columns.tolist()}")
print(f"Final shape: {master_df.shape}")
print(master_df.head())