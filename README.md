# CS506-Zillow-Home-Index-Forecasting-Model

## How to Build and Run the Code

To reproduce the current project pipeline, run:

```bash
make install
make clean-data
make features
make model
make test
```

If the cleaned and expanded datasets are already present, the shortest path is:

```bash
make install
make model
make test
```

### What each command does

- `make install`: installs project dependencies from `requirements.txt`
- `make clean-data`: rebuilds `data/processed/df_clean.csv`
- `make features`: rebuilds `data/processed/df_clean_with_all_features_model_ready_2010_2024.csv`
- `make model`: runs the model comparison pipeline and writes prediction/result files to `outputs/`
- `make test`: runs the test suite

## Supported Environment

- Python `3.10+`
- Tested in a Unix-like shell environment such as macOS or Linux

**Description**: Build a home price forecasting model using Zillow’s home index time series across U.S. metros and explain what drives home changes. By drives, we mean predictive importance, not causation. We will identify “drivers” as the features that measurably improve out-of-sample prediction and/or are consistently selected as important by the model. We will use Lasso penalization to determine this.

**Goal**: Predict next month/quarter home index value for each location with low error and rank the strongest drivers of home growth. We are not trying to compute or recreate ZORI using Zillow’s underlying index construction methodology. The goal is a standard out-of-sample forecasting task: using data available up to time t to predict future, not-yet-observed ZORI values at t+1. Essentially, we treat the published ZORI series as the target label, build a model/models that map predictors to future ZORI, and then evaluate performance on a time period using MAE/RMSE. Additionally, ZORI cannot be computed exactly from the historical ZORI series we download, because Zillow constructs it from unit-level repeat-home observations and applies reweighting and smoothing steps that require underlying listing data we don’t have. Our project is therefore to forecast future published ZORI values out-of-sample, not to reconstruct Zillow’s internal index calculation.

**Data collection**: Use Zillow Research CSVs and optionally merge public economic indicators like unemployment rate and home price index (rate of change in residential property prices) by geography and date.

**Modeling**: Compare Linear Regression, Ridge Regression, and Random Forest models on engineered housing and economic features to predict the next month’s home index value. We are not computing or reverse-engineering the index(ZORI) formula. We will treat the published home index series as the target label and build a model that forecasts future, unobserved months/quarters. The index may have a methodology, but future values are not computable today because they depend on future market conditions and data.

**Visualization**: Scatter Plot with a fitted regression showing the correlation of between Home Price Index (HPI) and Zillow Home Value Index (ZHVI). The points  mapped are based on same date, division, and unemployment rate.
The highest correlation came from the median income feature as the feature correlation score was 0.78. If you look at the heat map, you could see how much each feature we have extracted correlates to ZHVI.

<img width="659" height="567" alt="download" src="https://github.com/user-attachments/assets/3a5ca4bc-ab33-41ed-804e-f404e55c44f4" />

We can also distinguish the ZHVI trend by division to provide even more accurate results. More data visualization is available under 'data_visualization.ipynb'.
<img width="597" height="455" alt="download" src="https://github.com/user-attachments/assets/31722ed5-af79-4f6d-9496-2466941fc2a3" />

**Test plan**: Use time-based split and report MAE/RMSE

## Testing

The included tests focus on a few core parts of the pipeline:

- loading the expanded modeling dataset
- preprocessing data into model-ready features
- verifying that the time-based train/test split respects chronology
- confirming that model training and evaluation run end to end

Run tests with:

```bash
make test
```

## GitHub Workflow

GitHub Actions is configured to run the test suite automatically on pushes and pull requests.

The workflow:

- checks out the repository
- sets up Python
- installs dependencies
- runs `make test`

## Contributing

1. Create a branch for your changes.
2. Keep changes focused and documented.
3. Run `make test` before pushing.
4. Update the README if setup, usage, or outputs change.
