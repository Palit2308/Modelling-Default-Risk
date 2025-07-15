rule all:
    input: 
        # "Results/merton_model/notdropped/tables/two_system_summary_stats.csv",
        # "Results/merton_model/notdropped/tables/one_system_summary_stats.csv",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_PD_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Equity_Returns_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Equity_Vol_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Asset_Vol_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Implied_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Book_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Implied_A2D_vs_Book_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
        # "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Book_A2LTD_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_PD_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Equity_Returns_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Equity_Vol_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Asset_Vol_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Implied_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Book_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Implied_A2D_vs_Book_A2D_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
        # "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Book_A2LTD_Evolution.png"
        "Results/merton_model/dropna/plots/equity_asset_vol_trend.png",
        "Results/merton_model/notdropped/plots/equity_asset_vol_trend.png"
        # "Results/merton_model/notdropped/tables/two_system_merton_predictions.csv",
        # "Results/merton_model/notdropped/tables/two_system_merton_performance.csv",
        # "Results/merton_model/notdropped/tables/one_system_merton_predictions.csv",
        # "Results/merton_model/notdropped/tables/one_system_merton_performance.csv"
        # "Data/prepared_datasets/financials_annual_merton_imputed.csv",
        # "Results/merton_model/dropna/tables/two_system_summary_stats.csv",
        # "Results/merton_model/dropna/tables/one_system_summary_stats.csv",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_PD_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Equity_Returns_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Equity_Vol_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Asset_Vol_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Implied_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Book_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Implied_A2D_vs_Book_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
        # "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Book_A2LTD_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_PD_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Equity_Returns_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Equity_Vol_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Asset_Vol_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Implied_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Book_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Implied_A2D_vs_Book_A2D_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
        # "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Book_A2LTD_Evolution.png",
        # "Results/merton_model/dropna/tables/two_system_merton_predictions.csv",
        # "Results/merton_model/dropna/tables/two_system_merton_performance.csv",
        # "Results/merton_model/dropna/tables/one_system_merton_predictions.csv",
        # "Results/merton_model/dropna/tables/one_system_merton_performance.csv"
        # "Results/prepared_data_inspection/tables/cusip_level_null_values_percentage.csv",
        # "Results/prepared_data_inspection/plots/distribution_of_null_values_across_cusips.png",
        # "Results/prepared_data_inspection/plots/distribution_of_missingness_by_default_status.png",
        # "Results/prepared_data_inspection/tables/count_of_less_than_40_pct_null_by_default_status.csv"
        # "Results/prepared_data_inspection/tables/summary_table.csv",
        # "Results/prepared_data_inspection/tables/missing_at_distribution.csv",
        # "Results/prepared_data_inspection/plots/cusip_default_imbalance_at.png",
        # "Results/prepared_data_inspection/tables/missing_dt_distribution.csv",
        # "Results/prepared_data_inspection/plots/cusip_default_imbalance_dt.png",
        # "Results/prepared_data_inspection/tables/missing_ceq_distribution.csv",
        # "Results/prepared_data_inspection/plots/cusip_default_imbalance_ceq.png",
        # "Results/prepared_data_inspection/tables/missing_csho_distribution.csv",
        # "Results/prepared_data_inspection/plots/cusip_default_imbalance_csho.png",
        # "Results/prepared_data_inspection/plots/missingness_by_asset_quintiles_3A.png",
        # "Results/prepared_data_inspection/plots/missingness_extreme_values_3B.png",
        # "Results/prepared_data_inspection/plots/autocorrelation in variables.png",
        # "Results/prepared_data_inspection/plots/correlation_heatmap.png",
        # "Results/prepared_data_inspection/tables/data_default_summary_table.csv",
        # 'Results/prepared_data_inspection/plots/null_proportions_by_year.png',
        # "Data/prepared_datasets/stocks_data_prepared.csv"
        # "Data/prepared_datasets/financials_annual_merton.csv"
        # "Results/initial_inspection/no_of_defaults_per_year.csv"
        # "Results/initial_inspection/dominant_datadate_per_year.csv",
        # "Results/initial_inspection/companies_per_year_dominant_date.csv",
        # "Results/initial_inspection/defaulting_companies_year.csv",
        # "Results/initial_inspection/yearly_cases_default.csv",
        # "Results/initial_inspection/companies_with_gaps.csv",
        # "Results/initial_inspection/is_default_in_range.csv",
        # "Results/initial_inspection/how_many_in_vs_out_of_range.csv",
        # "Results/initial_inspection/out_of_range_distances.csv",
        # "Results/initial_inspection/in_range_distances.csv"

# rule imputed_merton_model_descriptives:
#     input:
#         "Results/merton_model/notdropped/tables/two_system_merton_predictions.csv",
#         "Results/merton_model/notdropped/tables/one_system_merton_predictions.csv"

#     output:
#         "Results/merton_model/notdropped/tables/two_system_summary_stats.csv",
#         "Results/merton_model/notdropped/tables/one_system_summary_stats.csv",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_PD_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Equity_Returns_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Equity_Vol_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Asset_Vol_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Implied_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Book_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Implied_A2D_vs_Book_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
#         "Results/merton_model/notdropped/plots/Two_System_Default_vs_Controls_Book_A2LTD_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_PD_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Equity_Returns_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Equity_Vol_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Asset_Vol_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Implied_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Book_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Implied_A2D_vs_Book_A2D_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
#         "Results/merton_model/notdropped/plots/One_System_Default_vs_Controls_Book_A2LTD_Evolution.png"

#     script: 
#         "Scripts/merton_model/model_checks_and_propensity_matching.py"

rule equtiy_asset_vol_tracking:
    input:
        "Results/merton_model/dropna/tables/two_system_merton_predictions.csv",
        "Results/merton_model/notdropped/tables/two_system_merton_predictions.csv"
    output:
        "Results/merton_model/dropna/plots/equity_asset_vol_trend.png",
        "Results/merton_model/notdropped/plots/equity_asset_vol_trend.png"
    script:
        "Scripts/merton_model/equity_vs_asset_volatility.py"

# rule merton_model_implimentation_imputed:
#     input:
#         "Data/prepared_datasets/financials_annual_merton_imputed.csv"

#     output:
#         "Results/merton_model/notdropped/tables/two_system_merton_predictions.csv",
#         "Results/merton_model/notdropped/tables/two_system_merton_performance.csv",
#         "Results/merton_model/notdropped/tables/one_system_merton_predictions.csv",
#         "Results/merton_model/notdropped/tables/one_system_merton_performance.csv"

#     script: 
#         "Scripts/merton_model/merton_model_implimentation_imputed.py"

# rule create_imputed_df:
#     input:
#         "Data/prepared_datasets/financials_annual_merton.csv",
#         "Data/prepared_datasets/stocks_data_prepared.csv",
#         "Data/daily_data/Yearly 1Y Treasury Rate.xlsx"
#     output:
#         "Data/prepared_datasets/financials_annual_merton_imputed.csv"
#     script:
#         "Scripts/imputed_data/create_imputed_data.py"

# rule merton_model_descriptives:
#     input:
#         "Results/merton_model/dropna/tables/two_system_merton_predictions.csv",
#         "Results/merton_model/dropna/tables/one_system_merton_predictions.csv"

#     output:
#         "Results/merton_model/dropna/tables/two_system_summary_stats.csv",
#         "Results/merton_model/dropna/tables/one_system_summary_stats.csv",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_PD_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Equity_Returns_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Equity_Vol_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Asset_Vol_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Implied_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Book_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Implied_A2D_vs_Book_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
#         "Results/merton_model/dropna/plots/Two_System_Default_vs_Controls_Book_A2LTD_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_PD_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Equity_Returns_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Equity_Vol_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Asset_Vol_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Implied_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Book_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Implied_A2D_vs_Book_A2D_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Implied_A2LTD_Evolution.png",
#         "Results/merton_model/dropna/plots/One_System_Default_vs_Controls_Book_A2LTD_Evolution.png"

#     script: 
#         "Scripts/merton_model/model_checks_and_propensity_matching.py"


# rule merton_model_implimentation:
#     input:
#         "Data/prepared_datasets/financials_annual_merton.csv",
#         "Data/prepared_datasets/stocks_data_prepared.csv",
#         "Data/daily_data/Yearly 1Y Treasury Rate.xlsx"

#     output:
#         "Results/merton_model/dropna/tables/two_system_merton_predictions.csv",
#         "Results/merton_model/dropna/tables/two_system_merton_performance.csv",
#         "Results/merton_model/dropna/tables/one_system_merton_predictions.csv",
#         "Results/merton_model/dropna/tables/one_system_merton_performance.csv"

#     script: 
#         "Scripts/merton_model/merton_model_implimentation.py"

# rule descriptive_statistics_2:
#     input:
#         "Data/prepared_datasets/financials_annual_merton.csv",
#         "Data/prepared_datasets/stocks_data_prepared.csv",
#         "Data/daily_data/Yearly 1Y Treasury Rate.xlsx"

#     output:
#         "Results/prepared_data_inspection/tables/cusip_level_null_values_percentage.csv",
#         "Results/prepared_data_inspection/plots/distribution_of_null_values_across_cusips.png",
#         "Results/prepared_data_inspection/plots/distribution_of_missingness_by_default_status.png",
#         "Results/prepared_data_inspection/tables/count_of_less_than_40_pct_null_by_default_status.csv"

#     script: 
#         "Scripts/descriptive_statistics/descriptive_statistics_2.py"


# rule descriptive_statistics_for_inspection:
#     input:
#         "Data/prepared_datasets/financials_annual_merton.csv",
#         "Data/prepared_datasets/stocks_data_prepared.csv",
#         "Data/daily_data/Yearly 1Y Treasury Rate.xlsx"

#     output:
#         "Results/prepared_data_inspection/tables/summary_table.csv",
#         "Results/prepared_data_inspection/tables/missing_at_distribution.csv",
#         "Results/prepared_data_inspection/plots/cusip_default_imbalance_at.png",
#         "Results/prepared_data_inspection/tables/missing_dt_distribution.csv",
#         "Results/prepared_data_inspection/plots/cusip_default_imbalance_dt.png",
#         "Results/prepared_data_inspection/tables/missing_ceq_distribution.csv",
#         "Results/prepared_data_inspection/plots/cusip_default_imbalance_ceq.png",
#         "Results/prepared_data_inspection/tables/missing_csho_distribution.csv",
#         "Results/prepared_data_inspection/plots/cusip_default_imbalance_csho.png",
#         "Results/prepared_data_inspection/plots/missingness_by_asset_quintiles_3A.png",
#         "Results/prepared_data_inspection/plots/missingness_extreme_values_3B.png",
#         "Results/prepared_data_inspection/plots/autocorrelation in variables.png",
#         "Results/prepared_data_inspection/plots/correlation_heatmap.png",
#         "Results/prepared_data_inspection/tables/data_default_summary_table.csv",
#         'Results/prepared_data_inspection/plots/null_proportions_by_year.png'

#     script: 
#         "Scripts/descriptive_statistics/descriptive_inspection.py"

# rule create_stocks_data:
#     input:
#         "Data/daily_data/stocks_daily_data.gz",
#         "Data/prepared_datasets/financials_annual_merton.csv"
#     output: 
#         "Data/prepared_datasets/stocks_data_prepared.csv"
#     script:
#         "Scripts/stock_data_handling/loading_stock_data.py"

# rule create_smaller_pandas_df:
#     input: 
#         "Data/financials/financials_annual.gz",
#         "Data/bankruptcy_filings/bankruptcy_filings_cleaned.csv"

#     output: 
#         "Data/prepared_datasets/financials_annual_merton.csv"

#     script: 
#         "Scripts/initial_inspection/create_short_df_annual.py"


# rule no_defaults_per_year_labelled:
#     input: 
#         "Data/financials/financials_annual.gz",
#         "Data/bankruptcy_filings/bankruptcy_filings_cleaned.csv"
# 
#     output: 
#         "Results/initial_inspection/no_of_defaults_per_year.csv"
# 
#     script: 
#         "Scripts/initial_inspection/no_defaults_per_year.py"

# rule summarising_default_years:
#     input: 
#         "Data/financials/financials_annual.gz",
#         "Results/initial_inspection/defaulting_companies_year.csv"
# 
#     output: 
#         "Results/initial_inspection/is_default_in_range.csv",
#         "Results/initial_inspection/how_many_in_vs_out_of_range.csv",
#         "Results/initial_inspection/out_of_range_distances.csv",
#         "Results/initial_inspection/in_range_distances.csv"
# 
#     script: 
#         "Scripts/initial_inspection/summarising_default_years.py"   

# rule inspect_defaulting_companies:
#     input: 
#         "Data/bankruptcy_filings/bankruptcy_filings_cleaned.csv",
#         "Data/financials/financials_annual.gz"
#     output: 
#         "Results/initial_inspection/defaulting_companies_year.csv",
#         "Results/initial_inspection/yearly_cases_default.csv"
#     script: 
#         "Scripts/initial_inspection/inspect_defaulting_companies.py"

# rule checking_gaps:
#     input: 
#         "Data/financials/financials_annual.gz"
#     output: 
#         "Results/initial_inspection/companies_with_gaps.csv"
#     script: 
#         "Scripts/initial_inspection/checking_gaps.py"

# rule initial_panel_inspection:
#     input: 
#         "Data/financials/financials_annual.gz"
#     output: 
#         "Results/initial_inspection/dominant_datadate_per_year.csv",
#         "Results/initial_inspection/companies_per_year_dominant_date.csv"
#     script: 
#         "Scripts/initial_inspection/inspect_panel_structure.py"









