import pandas as pd

# cleaning zhvi.csv file
zhvi = pd.read_csv("data/zhvi.csv")

# understand the structure
zhvi = zhvi.loc[:, ~zhvi.columns.str.contains('^Unnamed')]
zhvi = zhvi.dropna(how='all')  # drop completely empty rows

# remove columns
zhvi.columns = zhvi.columns.str.strip().str.lower().str.replace(" ", "_")

# identify date columns (they start with a year like "2000-01-31")
date_cols = [col for col in zhvi.columns if col[:4].isdigit()]

# everything else is an id column
id_cols = [col for col in zhvi.columns if col not in date_cols]

zhvi_long = zhvi.melt(
    id_vars=id_cols,
    value_vars=date_cols,   # 👈 IMPORTANT (prevents mistakes)
    var_name="date",
    value_name="zhvi"
)

# convert date safely
zhvi_long["date"] = pd.to_datetime(zhvi_long["date"], format="%Y-%m-%d")

# sort
zhvi_long = zhvi_long.sort_values(by=["regionname", "date"])

# print(zhvi_long.head())

zhvi_long["zhvi"] = zhvi_long.groupby("regionname")["zhvi"].transform(
    lambda x: x.interpolate()
)

print(zhvi.head())
# print(zhvi.columns)



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