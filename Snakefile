rule all:
    input:
        "results_quarterly/plots/PD by quality rating.png",
        "results_quarterly/plots/PD with time.png",
        "results_quarterly/plots/Eq and Asset sigma with time.png",
        "results_quarterly/tables/number_of_defaults_per_year.csv",
        "results_quarterly/plots/quarterly total negative biod news.png"



# rule prepare_quarterly_data:
#     input:
#         "Data/Raw Data/fundamentals_quarterly.gz",
#         "Data\Raw Data\Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023\Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.xlsx",
#         "Data/Raw Data/return_volatility_df.csv"
#     output:
#         "Data/Prepared Data/prepared_data_merton_pd.csv",
#         "results_quarterly/tables/model_performance.csv"
#     script:
#         "python_scripts/scripts/quarterly_data_preparation.py"

rule descriptive_statistics:
    input:
        "Data/Prepared Data/prepared_data_merton_pd.csv",
        "Data/Raw Data/nyt_indices.csv"
    output:
        "results_quarterly/plots/PD by quality rating.png",
        "results_quarterly/plots/PD with time.png",
        "results_quarterly/plots/Eq and Asset sigma with time.png",
        "results_quarterly/tables/number_of_defaults_per_year.csv",
        "results_quarterly/plots/quarterly total negative biod news.png"
    script:
        "python_scripts/scripts/descriptive_statistics.py"

# rule prepare_biodiversity_data:
#     input:
#         "Data/Prepared Data/prepared_data_merton_pd.csv",
#         "Data/Raw Data/10k_biodiversity_scores.csv",
#         "Data/Raw Data/nyt_indices.csv",
#         "Data/Raw Data/survey_biodiversity_scores.csv"

#     output:
#         "Data/prepared_data_regression.csv",
#         "results_quarterly/plots/Av Bio Risk by Gics4.png",
#         "results_quarterly/plots/Neg Bio Mentions by Gics4.png",
#         "results_quarterly/tables/correlation_biodiversity.csv",
#         "results_quarterly/tables/reg_results_nofe.csv",
#         "results_quarterly/tables/reg_nofe_r2.csv",
#         "results_quarterly/plots/QQ Plot Nofe.png",
#         "results_quarterly/tables/reg_results_fe.csv",
#         "results_quarterly/tables/reg_fe_r2.csv",
#         "results_quarterly/plots/QQ Plot Fe.png",
#         "results_quarterly/tables/vif_neg_av_risk_news.csv",
#         "results_quarterly/tables/error_diagnostics.csv",
#         "results_quarterly/tables/vif_neg_av_risk_news_with_lag.csv",
#         "results_quarterly/tables/reg_results_fe_lags.csv",
#         "results_quarterly/tables/reg_fe_lags_r2.csv",
#         "results_quarterly/plots/QQ Plot Fe lags.png"

#     script:
#         "python_scripts/scripts/biodiversity_variables.py"