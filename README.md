# Modelling Default Risk

This repository builds a firm-level default-risk dataset and applies structural credit-risk methods based on the Merton model. It combines accounting statements, daily stock prices, bankruptcy filings, and biodiversity-related indicators to estimate probabilities of default and study how firm risk evolves over time.

The workflow is organised with **Snakemake**. Large raw financial and market-data transformations are performed with **PySpark**, while smaller model-ready datasets are converted to pandas for estimation, diagnostics, tables, and plots.

## Data pipeline

1. Inspect and clean the annual firm panel.
2. Match bankruptcy filings to firms using CUSIP identifiers and construct default labels.
3. Process daily stock prices into annual firm-level returns and volatility.
4. Merge market and accounting variables required by the Merton model.
5. Estimate default probabilities and evaluate model performance.
6. Produce descriptive results and empirical analyses using quarterly data and biodiversity-risk measures.

## Spark-based processing on a virtual server

The data-intensive scripts are designed to run on a virtual server with Apache Spark. After connecting to the server—for example through SSH—the workflow creates a `SparkSession` using:

```python
SparkSession.builder.appName("US1").getOrCreate()
```

The scripts therefore inherit the Spark configuration of the environment in which they are launched. When Spark is configured to use the virtual server's available cores, it partitions the underlying data and executes DataFrame transformations in parallel. Operations such as filtering, joins, grouped aggregations, and window calculations can consequently be distributed across cores instead of being processed row by row on a single local process.

For a single multi-core virtual machine, Spark can be configured to use all available cores with `local[*]`. The server connection itself is external to the Python scripts; Spark manages computation after the workflow has been started on the server.

## `Scripts/stock_data_handling`

`loading_stock_data.py` converts the large daily stock-price file into annual firm-level market measures required by the default-risk model. It:

- loads daily observations as a Spark DataFrame;
- retains firms appearing in the annual financial dataset;
- partitions observations by CUSIP and year;
- forward-fills price, adjustment-factor, and total-return-factor fields within each firm-year window;
- constructs an adjusted daily return price and lagged log returns; and
- aggregates daily returns into annual expected returns and equity volatility.

The expensive window operations and firm-year aggregations remain in Spark. Only the much smaller aggregated result is converted to pandas and written to CSV.

## `Scripts/initial_inspection`

These scripts validate the raw annual panel and bankruptcy labels before model estimation:

- `inspect_panel_structure.py` detects duplicate CUSIP-year observations, identifies dominant reporting dates, and retains the latest applicable filing.
- `checking_gaps.py` finds firms with missing years between their first and last observations.
- `inspect_defaulting_companies.py` filters Chapter 7 and Chapter 11 filings, matches them to financial records by CUSIP, and summarises defaults by year.
- `summarising_default_years.py` checks whether recorded bankruptcy years fall inside each firm's observed financial history.
- `create_short_df_annual.py` creates a smaller labelled dataset containing the variables needed for Merton estimation.
- `no_defaults_per_year.py` reports annual observations and newly identified default events.

These tasks rely mainly on Spark joins, grouped summaries, and partitioned window functions, allowing large firm-year panels to be checked efficiently across the virtual server's processing cores.

## Repository structure

```text
Scripts/
├── stock_data_handling/   # Distributed daily stock-data preparation
├── initial_inspection/    # Panel, gap, bankruptcy, and label checks
├── functions/             # Reusable cleaning, matching, and model functions
├── merton_model/          # Merton estimation and model evaluation
└── tests/                 # Data-labelling and panel-structure checks

python_scripts/            # Quarterly preparation, analysis, and regressions
Results/                   # Annual-pipeline outputs
results_quarterly/         # Quarterly tables and figures
Snakefile                  # Reproducible workflow definition
```

## Main technologies

Python, PySpark, pandas, NumPy, SciPy, scikit-learn, statsmodels, Matplotlib, Seaborn, and Snakemake.

> Raw financial, market, and bankruptcy datasets are not included in the repository and must be supplied separately in the paths expected by the Snakemake rules.
