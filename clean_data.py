import pandas as pd

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

# -------------------------------
# 2. Load + clean ZHVI
# -------------------------------
zhvi = pd.read_csv("data/zhvi.csv")

zhvi = zhvi.loc[:, ~zhvi.columns.str.contains('^Unnamed')]
zhvi = zhvi.dropna(how='all')

zhvi.columns = zhvi.columns.str.strip().str.lower().str.replace(" ", "_")

# -------------------------------
# 3. Reshape (wide → long)
# -------------------------------
date_cols = [col for col in zhvi.columns if col[:4].isdigit()]
id_cols = [col for col in zhvi.columns if col not in date_cols]

zhvi_long = zhvi.melt(
    id_vars=id_cols,
    value_vars=date_cols,
    var_name="date",
    value_name="zhvi"
)

# -------------------------------
# 4. Clean + format
# -------------------------------
zhvi_long["date"] = pd.to_datetime(zhvi_long["date"], format="%Y-%m-%d")

# keep only metro areas (better consistency)
zhvi_long = zhvi_long[zhvi_long["regiontype"] == "msa"]

# drop bad rows
zhvi_long = zhvi_long.dropna(subset=["statename"])

# sort before interpolation
zhvi_long = zhvi_long.sort_values(by=["regionname", "date"])

# interpolate missing values
zhvi_long["zhvi"] = zhvi_long.groupby("regionname")["zhvi"].transform(
    lambda x: x.interpolate()
)

# -------------------------------
# 5. Map to Census divisions
# -------------------------------
zhvi_long["division"] = zhvi_long["statename"].map(state_to_division)

# drop anything unmapped (safety)
zhvi_long = zhvi_long.dropna(subset=["division"])

# -------------------------------
# 6. Aggregate → division level
# -------------------------------
zhvi_division = zhvi_long.groupby(["division", "date"])["zhvi"].mean().reset_index()

# -------------------------------
# 7. Final output
# -------------------------------
print(zhvi_division.head())
print(zhvi_division.shape)



# cleaning hpi file
hpi = pd.read_excel("data/hpi.xlsx", skiprows=1)
# understand the structure
hpi = hpi.loc[:, ~hpi.columns.str.contains('^Unnamed')]
hpi = hpi.dropna(how='all')  # drop completely empty rows
# remove columns
hpi.columns = hpi.columns.str.strip().str.lower().str.replace(" ", "_")
# convert data types

# HPI specific




print(hpi.head())
print(hpi.columns)





# # # cleaning upi file
# # upi = pd.read_excel("data/upi.xlsx", skiprows=2)
# # print(upi.head(20))
# # # understand the structure
# # upi = upi.loc[:, ~upi.columns.str.contains('^Unnamed')]
# # upi = upi.dropna(how='all')  # drop completely empty rows
# # # remove columns
# # upi.columns = upi.columns.str.strip().str.lower().str.replace(" ", "_")

# # # print(upi.head())
# # # print(upi.columns)