import pandas as pd

# cleaning zhvi.csv file
zhvi = pd.read_csv("data/zhvi.csv")
# understand the structure
zhvi = zhvi.loc[:, ~zhvi.columns.str.contains('^Unnamed')]
zhvi = zhvi.dropna(how='all')  # drop completely empty rows
# remove columns
zhvi.columns = zhvi.columns.str.strip().str.lower().str.replace(" ", "_")
id_cols = ["regionid", "regionname", "statename"]

zhvi_long = zhvi.melt(
    id_vars=id_cols,
    var_name="date",
    value_name="zhvi"
)

# convert date
zhvi_long["date"] = pd.to_datetime(zhvi_long["date"])

# sort
zhvi_long = zhvi_long.sort_values(by=["regionname", "date"])

print(zhvi_long.head())

zhvi_long["zhvi"] = zhvi_long.groupby("regionname")["zhvi"].transform(
    lambda x: x.interpolate()
)

print(zhvi.head())
print(zhvi.columns)



# cleaning hpi file
hpi = pd.read_excel("data/hpi.xlsx", skiprow=1)
# understand the structure
hpi = hpi.loc[:, ~hpi.columns.str.contains('^Unnamed')]
hpi = hpi.dropna(how='all')  # drop completely empty rows
# remove columns
hpi.columns = hpi.columns.str.strip().str.lower().str.replace(" ", "_")
# convert data types

# HPI specific




print(hpi.head())
print(hpi.columns)





# # cleaning upi file
# upi = pd.read_excel("data/upi.xlsx", skiprows=2)
# print(upi.head(20))
# # understand the structure
# upi = upi.loc[:, ~upi.columns.str.contains('^Unnamed')]
# upi = upi.dropna(how='all')  # drop completely empty rows
# # remove columns
# upi.columns = upi.columns.str.strip().str.lower().str.replace(" ", "_")

# # print(upi.head())
# # print(upi.columns)