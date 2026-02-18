# CS506-Zillow-Rent-Index-Forecasting-Model

**Description**: Build a rent price forecasting model using Zillow’s rental index time series across U.S. metros and explain what drives rent changes. By drives, we mean predictive importance, not causation. We will identify “drivers” as the features that measurably improve out-of-sample prediction and/or are consistently selected as important by the model. We will use Lasso penalization to determine this. 


**Goal**: Predict next month/quarter rent index value for each location with low error and rank the strongest drivers of rent growth. We are not trying to compute or recreate ZORI using Zillow’s underlying index construction methodology. The goal is a standard out-of-sample forecasting task: using data available up to time t to predict future, not-yet-observed ZORI values at t+1. Essentially, we treat the published ZORI series as the target label, build a model/models that map predictors to future ZORI, and then evaluate performance on a time period using MAE/RMSE. Additionally, ZORI cannot be computed exactly from the historical ZORI series we download, because Zillow constructs it from unit-level repeat-rent observations and applies reweighting and smoothing steps that require underlying listing data we don’t have. Our project is therefore to forecast future published ZORI values out-of-sample, not to reconstruct Zillow’s internal index calculation.

**Data collection**: Use Zillow Research CSVs and optionally merge public economic indicators by geography and date.

**Modeling**: Use XGBoost regression with lag + rolling-window features from the rent index time series to predict the next month’s rent index value. We are not computing or reverse-engineering the index(ZORI) formula. We will treat the published rent index series as the target label and build a model that forecasts future, unobserved months/quarters. The index may have a methodology, but future values are not computable today because they depend on future market conditions and data.

**Visualization**: TBD

**Test plan**: Use time-based split and report MAE/RMSE
