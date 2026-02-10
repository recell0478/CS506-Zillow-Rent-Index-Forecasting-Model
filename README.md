# CS506-Zillow-Rent-Index-Forecasting-Model

Description: Build a rent price forecasting model using Zillow’s rental index time series across U.S. metros and explain what drives rent changes.

Goal: Predict next month/quarter rent index value for each location with low error and rank the strongest drivers of rent growth.

Data collection: Use Zillow Research CSVs and optionally merge public economic indicators by geography and date.

Modeling: Use XGBoost regression with lag + rolling-window features from the rent index time series to predict the next month’s rent index value.
