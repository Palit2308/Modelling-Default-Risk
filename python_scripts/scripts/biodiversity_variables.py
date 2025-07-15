import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.stattools import durbin_watson

df = pd.read_csv("Data/Prepared Data/prepared_data_merton_pd.csv") #snakemake.input[0]

biod_score = pd.read_csv("Data/Raw Data/10k_biodiversity_scores.csv") #snakemake.input[1]

biod_score = biod_score[["cusip", "year" , "count", "negative", "regulation"]]

biod_score = biod_score.rename(columns={"year": "fyearq"})

df_merged = df.merge(
    biod_score, 
    on=['cusip', 'fyearq'], 
    how='left'
)

nyt_scores = pd.read_csv("Data/Raw Data/nyt_indices.csv") #snakemake.input[12

nyt_scores['date'] = pd.to_datetime(nyt_scores['date'], format='%d%b%Y')

nyt_scores['fyearq'] = nyt_scores['date'].dt.year
nyt_scores['fqtr'] = nyt_scores['date'].dt.quarter

biodiv_qtr_sum = nyt_scores.groupby(['fyearq', 'fqtr'])['biodiversity'].sum().reset_index()

biodiv_qtr_sum = biodiv_qtr_sum.rename(columns={'biodiversity': 'net_bad_minus_good_news'})

df_merged = df_merged.merge(
    biodiv_qtr_sum, 
    on=['fyearq', 'fqtr'], 
    how='left'
)


survey_biod = pd.read_csv("Data/Raw Data/survey_biodiversity_scores.csv") #snakemake.input[3]

survey_biod = survey_biod.rename(columns={
    'transition': 'transition_risk',
    'physical': 'physical_risk',
    'average': 'average_risk'
})

df_merged['gics4'] = df_merged['gind'].astype(str).str[:4]

df_merged['gics4'] = df_merged['gics4'].astype(str)
survey_biod['gics4'] = survey_biod['gics4'].astype(str)


df_merged = df_merged.merge(
    survey_biod,
    on='gics4',
    how='left'
)


df_merged.describe().T

#######################################################################################################################################

df_final = df_merged.dropna()

df_final.to_csv("Data/Prepared Data/prepared_data_regression.csv", index = False) #, snakemake.output[0]

#######################################################################################################################################

# Plotting Average Biodiversity risk by Industry

avg_risk_by_gics4 = (
    df_final.groupby('gics4')['average_risk']
    .mean()
    .reset_index()
    .sort_values('average_risk', ascending=False)
)

# Plot
plt.figure(figsize=(10,5))
plt.bar(avg_risk_by_gics4['gics4'], avg_risk_by_gics4['average_risk'])
plt.xlabel('GICS4')
plt.ylabel('Average Biodiversity Risk')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_quarterly/plots/Av Bio Risk by Gics4.png", dpi=300) #, snakemake.output[1]

#######################################################################################################################################

# Net Negative Biodiversity Mentions in the sample, per industry

neg_by_gics4 = (
    df_final.groupby('gics4')['negative']
    .sum()
    .reset_index()
    .sort_values('negative', ascending=False)
)

# Plot
plt.figure(figsize=(10,5))
plt.bar(neg_by_gics4['gics4'], neg_by_gics4['negative'])
plt.xlabel('GICS4')
plt.ylabel('Negative Biodiversity Mentions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_quarterly/plots/Neg Bio Mentions by Gics4.png", dpi=300) # snakemake.output[2]


#######################################################################################################################################

# Correlation study between the different measures of risk and PD

df_risk = df_final[["PD", "count", "regulation", "negative", "net_bad_minus_good_news", "transition_risk", "physical_risk", "average_risk"]]

correlation_df  = df_risk.corr()

correlation_df.to_csv("results_quarterly/tables/correlation_biodiversity.csv", index = False) #snakemake.output[3]


#######################################################################################################################################

df_final.describe().T

#######################################################################################################################################

# Model without FE

results = smf.ols(
    'PD ~ negative + average_risk * net_bad_minus_good_news', 
    data=df_final
).fit(cov_type='cluster', cov_kwds={'groups': df_final['cusip']}, use_t=True)

# Get the summary table
summary_table = results.summary2().tables[1]

# Filter out all fixed effects terms (they start with 'C(cusip)' or 'C(fyearq)')
filtered_summary = summary_table[~summary_table.index.str.startswith('C(cusip)') & 
                                 ~summary_table.index.str.startswith('C(fyearq)')]

print(filtered_summary)

r2_nofe = pd.DataFrame({'R_squared NO FE': [results.rsquared]})

filtered_summary.to_csv("results_quarterly/tables/reg_results_nofe.csv") #snakemake.output[4]
r2_nofe.to_csv("results_quarterly/tables/reg_nofe_r2.csv", index = False) #snakemake.output[5]

df_final['PD_fitted_nofe'] = results.fittedvalues
df_final['residuals_nofe'] = results.resid

# QQ Plot of residuals

sm.qqplot(df_final['residuals_nofe'], line='45')
plt.tight_layout()
plt.savefig("results_quarterly/plots/QQ Plot Nofe.png", dpi=300) #snakemake.output[6]

#######################################################################################################################################

# model with FE

results1 = smf.ols(
    'PD ~ negative + average_risk * net_bad_minus_good_news + C(cusip) + C(fyearq)', 
    data=df_final
).fit(cov_type='cluster', cov_kwds={'groups': df_final['cusip']}, use_t=True)

# Get the summary table
summary_table1 = results1.summary2().tables[1]

# Filter out all fixed effects terms (they start with 'C(cusip)' or 'C(fyearq)')
filtered_summary1 = summary_table1[~summary_table1.index.str.startswith('C(cusip)') & 
                                 ~summary_table1.index.str.startswith('C(fyearq)')]

print(filtered_summary1)

print(results1.rsquared)

r2_fe = pd.DataFrame({'R_squared FE': [results1.rsquared]})

filtered_summary1.to_csv("results_quarterly/tables/reg_results_fe.csv") #snakemake.output[7]
r2_fe.to_csv("results_quarterly/tables/reg_fe_r2.csv", index = False) #snakemake.output[8]

df_final['PD_fitted_fe'] = results1.fittedvalues
df_final['residuals_fe'] = results1.resid

# QQ Plot of residuals

sm.qqplot(df_final['residuals_fe'], line='45')
plt.tight_layout()
plt.savefig("results_quarterly/plots/QQ Plot Fe.png", dpi=300) #snakemake.output[9]

#######################################################################################################################################

# Multicollinearity check

X = df_final[['negative', 'average_risk', 'net_bad_minus_good_news']]

X = sm.add_constant(X)

vif_df = pd.DataFrame()
vif_df['feature'] = X.columns
vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_df.to_csv("results_quarterly/tables/vif_neg_av_risk_news.csv", index = False) #snakemake.output[10]


# Check of normality of residuals, skewness and kurtosis

jb_stat, jb_pvalue = jarque_bera(results1.resid)
skewness = skew(results1.resid)
kurt = kurtosis(results1.resid)
dw_stat = durbin_watson(results1.resid)

print(f'Durbin-Watson statistic: {dw_stat:.3f}')
print(f'Jarque-Bera test statistic: {jb_stat}, p-value: {jb_pvalue}')
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurt}')

jb_dict = {
    'Jarque-Bera Statistic': [jb_stat],
    'p-value': [jb_pvalue],
    'Skewness': [skewness],
    'Kurtosis': [kurt],
    'DW Stat': [dw_stat]
}

jb_df = pd.DataFrame(jb_dict)

jb_df.to_csv("results_quarterly/tables/error_diagnostics.csv", index = False)  #snakemake.output[11]


#######################################################################################################################################

df_final_1 = df_final.copy()

df_final_1 = df_final_1.sort_values(['cusip', 'fyearq', 'fqtr'])  # Ensure sorted by firm and time
df_final_1['PD_lag'] = df_final_1.groupby('cusip')['PD'].shift(1)

df_final_1 = df_final_1.dropna()

# Multicollinearity check

X1 = df_final_1[['negative', 'average_risk', 'net_bad_minus_good_news', 'PD_lag']]

X1 = sm.add_constant(X1)

vif_df = pd.DataFrame()
vif_df['feature'] = X1.columns
vif_df['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif_df.to_csv("results_quarterly/tables/vif_neg_av_risk_news_with_lag.csv", index = False) #snakemake.output[12]


# model with lags

results2 = smf.ols(
    'PD ~ negative + average_risk * net_bad_minus_good_news + PD_lag + C(cusip) + C(fyearq)', 
    data=df_final_1
).fit(cov_type='cluster', cov_kwds={'groups': df_final_1['cusip']}, use_t=True)

# Get the summary table
summary_table2 = results2.summary2().tables[1]

# Filter out all fixed effects terms (they start with 'C(cusip)' or 'C(fyearq)')
filtered_summary2 = summary_table2[~summary_table2.index.str.startswith('C(cusip)') & 
                                 ~summary_table2.index.str.startswith('C(fyearq)')]

print(filtered_summary2)

print(results2.rsquared)

r2_fe_lags = pd.DataFrame({'R_squared FE': [results2.rsquared]})

filtered_summary1.to_csv("results_quarterly/tables/reg_results_fe_lags.csv") #snakemake.output[13]

r2_fe_lags.to_csv("results_quarterly/tables/reg_fe_lags_r2.csv", index = False) #snakemake.output[14]

df_final_1['PD_fitted_fe_lags'] = results2.fittedvalues
df_final_1['residuals_fe_lags'] = results2.resid

# QQ Plot of residuals

sm.qqplot(df_final_1['residuals_fe_lags'], line='45')
plt.tight_layout()
plt.savefig("results_quarterly/plots/QQ Plot Fe lags.png", dpi=300) #snakemake.output[15]







df_final_1 = df_final.copy()
df_final_1 = df_final_1.sort_values(['cusip', 'fyearq', 'fqtr'])  # Ensure sorted by firm and time
df_final_1['PD_lag'] = df_final_1.groupby('cusip')['PD'].shift(1)
df_final_1['gdwlq_pct_change'] = df_final_1.groupby('cusip')['gdwlq'].pct_change()
df_final_1.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final_1 = df_final_1.dropna()
df_final_1.describe().T

X2 = df_final_1[['negative', 'average_risk', 'net_bad_minus_good_news', 'PD_lag', 'gdwlq', 'gdwlq_pct_change']]

X2 = sm.add_constant(X2)

vif_df = pd.DataFrame()
vif_df['feature'] = X2.columns
vif_df['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]







df_final_1 = df_final.copy()
df_final_1 = df_final_1.sort_values(['cusip', 'fyearq', 'fqtr'])  # Ensure sorted by firm and time
df_final_1['PD_lag'] = df_final_1.groupby('cusip')['PD'].shift(1)
df_final_1['PD_change'] = df_final_1['PD'] - df_final_1['PD_lag']
df_final_1['PD_change_lag'] = df_final_1['PD_change'].shift()
df_final_1 = df_final_1.dropna()
df_final_1.describe().T

results3 = smf.ols(
    'PD_change ~ negative + average_risk * net_bad_minus_good_news + PD_lag + PD_change_lag + C(fyearq)', 
    data=df_final_1
).fit(cov_type='cluster', cov_kwds={'groups': df_final_1['cusip']}, use_t=True)

# Get the summary table
summary_table3 = results3.summary2().tables[1]

# Filter out all fixed effects terms (they start with 'C(cusip)' or 'C(fyearq)')
filtered_summary3 = summary_table3[~summary_table3.index.str.startswith('C(cusip)') & 
                                 ~summary_table3.index.str.startswith('C(fyearq)')]

print(filtered_summary3)

print(results3.rsquared)

# Assuming results3 is already fitted and df_final_1 is your DataFrame
df_final_1['PD_fitted_fe_lags'] = results3.fittedvalues
df_final_1['residuals_fe_lags'] = results3.resid

# Estimate degrees of freedom for the t-distribution
df_t = results3.df_resid  # Or simply use: df_t = len(df_final_1) - number_of_params

# QQ Plot of residuals vs t-distribution
sm.qqplot(df_final_1['residuals_fe_lags'], dist=stats.t, distargs=(df_t,), line='45')

plt.title(f"QQ-Plot of Residuals vs t-Distribution (df={df_t:.0f})")
plt.tight_layout()
plt.show()
plt.savefig("results_quarterly/plots/QQ Plot Fe lags.png", dpi=300) #snakemake.output[15]




from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# assuming `results` is your fitted model, e.g. from smf.ols(...).fit()
resid = results3.resid

# make your plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(resid, lags=20, ax=axes[0], title="ACF of Residuals")
plot_pacf(resid, lags=20, ax=axes[1], title="PACF of Residuals")
plt.tight_layout()
plt.show()












# # Fit quantile regression at the median (q=0.5)
# quant_reg = smf.quantreg(
#     'PD ~ negative + average_risk * net_bad_minus_good_news + C(cusip) + C(fyearq)', 
#     data=df_final
# )
# results_q10 = quant_reg.fit(q=0.1)

# # Get the summary table
# summary_table_q10 = results_q10.summary2().tables[1]

# # Filter out all fixed effects terms
# filtered_summary_q10 = summary_table_q10[
#     ~summary_table_q10.index.str.startswith('C(cusip)') & 
#     ~summary_table_q10.index.str.startswith('C(fyearq)')
# ]

# print(filtered_summary_q10)
# print(results_q10.rsquared)


df_final.describe().T

# from sklearn.linear_model import QuantileRegressor

# # Select features and target
# features = ['negative', 'average_risk', 'net_bad_minus_good_news', 'cusip', 'fyearq']
# X = df_final[features]
# y = df_final['PD']

# # One-hot encode categorical variables (cusip, fyearq)
# X_encoded = pd.get_dummies(X, columns=['cusip', 'fyearq'], drop_first=True)

# # Fit quantile regression for median (q=0.5)
# qr = QuantileRegressor(quantile=0.1, alpha=0, solver='highs')
# qr.fit(X_encoded, y)

# print("Intercept:", qr.intercept_)
# print("Coefficients:", qr.coef_)



# # Run quantile regression at the 50th percentile (median)
# quant_reg = smf.quantreg(
#     'PD ~ negative + average_risk * net_bad_minus_good_news + C(cusip) + C(fyearq)',
#     data=df_final
# )
# result = quant_reg.fit(q=0.1)

# # Get the summary table (includes Coef., Std.Err., t, and P>|t|)
# summary_table = result.summary2().tables[1]

# # Remove fixed effects if desired
# filtered_summary = summary_table[
#     ~summary_table.index.str.startswith('C(cusip)') & 
#     ~summary_table.index.str.startswith('C(fyearq)')
# ]
# print(filtered_summary)