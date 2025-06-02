import pandas as pd
import matplotlib.pyplot as plt
from Scripts.functions.handling_prepared_data.joining_stocks_firms_data import join_stocks_firms_data
import numpy as np
import seaborn as sns
import statsmodels.api as sm
###############################################################################################################
input1 = snakemake.input[0]
input2 = snakemake.input[1]
input3 = snakemake.input[2]

df_annual = pd.read_csv(input1)
df_stocks = pd.read_csv(input2)
df_interest = pd.read_excel(input3)

df = join_stocks_firms_data(df_stocks, df_annual, df_interest)


summary = df.describe(include='all')
summary.to_csv(snakemake.output[0])
###############################################################################################################
# MISSING ASSET VALUE - MORE DEFAULT FIRMS OR MORE NON DEFAULT FIRMS

cusips_with_missing_at = df[df['at'].isna()].groupby('cusip').size().reset_index(name='missing_at_count')

cusip_default_flags = df.groupby('cusip')['default_status'].apply(lambda x: (x == 'D').any()).reset_index(name='ever_defaulted')

summary = cusips_with_missing_at.merge(cusip_default_flags, on='cusip', how='left')

summary.to_csv(snakemake.output[1])

counts = summary['ever_defaulted'].value_counts().rename(index={True: 'Defaulted', False: 'Not Defaulted'})

plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['tomato', 'skyblue'])
plt.title('CUSIP Default Imbalance')
plt.ylabel('Number of CUSIPs')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(snakemake.output[2], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################
# MISSING DEBT VALUE - MORE DEFAULTED FIRMS OR MORE NON DEFAULTED FIRMS

cusips_with_missing_dt = df[df['dt'].isna()].groupby('cusip').size().reset_index(name='missing_at_count')

cusip_default_flags = df.groupby('cusip')['default_status'].apply(lambda x: (x == 'D').any()).reset_index(name='ever_defaulted')

summary = cusips_with_missing_dt.merge(cusip_default_flags, on='cusip', how='left')

summary.to_csv(snakemake.output[3])

counts = summary['ever_defaulted'].value_counts().rename(index={True: 'Defaulted', False: 'Not Defaulted'})

plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['tomato', 'skyblue'])
plt.title('CUSIP Default Imbalance')
plt.ylabel('Number of CUSIPs')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(snakemake.output[4], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################
# MISSING COMMON EQUITY VALUE - MORE DEFAULTED FIRMS OR MORE NON DEFAULTED FIRMS

cusips_with_missing_ceq= df[df['ceq'].isna()].groupby('cusip').size().reset_index(name='missing_at_count')

cusip_default_flags = df.groupby('cusip')['default_status'].apply(lambda x: (x == 'D').any()).reset_index(name='ever_defaulted')

summary = cusips_with_missing_ceq.merge(cusip_default_flags, on='cusip', how='left')

summary.to_csv(snakemake.output[5])

counts = summary['ever_defaulted'].value_counts().rename(index={True: 'Defaulted', False: 'Not Defaulted'})

plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['tomato', 'skyblue'])
plt.title('CUSIP Default Imbalance')
plt.ylabel('Number of CUSIPs')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(snakemake.output[6], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################
# MISSING SHARES OUTSTANDING VALUE - MORE DEFAULTED FIRMS OR MORE NON DEFAULTED FIRMS

cusips_with_missing_csho= df[df['csho'].isna()].groupby('cusip').size().reset_index(name='missing_at_count')

cusip_default_flags = df.groupby('cusip')['default_status'].apply(lambda x: (x == 'D').any()).reset_index(name='ever_defaulted')

summary = cusips_with_missing_ceq.merge(cusip_default_flags, on='cusip', how='left')

summary.to_csv(snakemake.output[7])

counts = summary['ever_defaulted'].value_counts().rename(index={True: 'Defaulted', False: 'Not Defaulted'})

# mean_missing_by_default_status = summary.groupby('ever_defaulted')['missing_at_count'].mean()
# mean_missing_by_default_status.index = mean_missing_by_default_status.index.map({True: 'Defaulted', False: 'Not Defaulted'})
# print(mean_missing_by_default_status)

plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['tomato', 'skyblue'])
plt.title('CUSIP Default Imbalance')
plt.ylabel('Number of CUSIPs')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(snakemake.output[8], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################
# MISSINGNESS BY ASSET QUINTILES

df_clean = df[df['at'].notna()].copy()

df_clean['size_quintile'] = pd.qcut(df_clean['at'], q=5, labels=False) + 1

characteristics = ['ch', 'ceq', 'csho', 'lct', 'dd1', 'dltt', 'dt',
                    'gp', 'ebit', 'sale', 'default_status', 'prcc_c', 'expected_return',
                    'volatility']

missing_by_quintile = df_clean.groupby('size_quintile')[characteristics].apply(lambda x: x.isna().mean() * 100)
missing_by_quintile['average_missing'] = missing_by_quintile.mean(axis=1)

plt.figure(figsize=(10, 6))

for col in characteristics:
    plt.plot(missing_by_quintile.index, missing_by_quintile[col], linestyle='--', alpha=0.7, label=col)

plt.plot(missing_by_quintile.index, missing_by_quintile['average_missing'], color='black', linewidth=2.5, label='Average (Black Line)')

plt.xlabel('Size Quintile (1 = Smallest, 5 = Largest)')
plt.ylabel('% Missing Values')
plt.title('Missingness by Firm Size Quintiles')
plt.legend(loc='best', fontsize='small', frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[9], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################

# MISSINGNESS EXTREME VALUES - 3B
data = df.copy()
variables = ['expected_return', 'volatility', 'ceq', 'csho']  # replace with full list

filtered_data = {}
avg_per_cusip = {}
cusip_quintiles = {}
missingness_by_quintile = {}
for col in variables:

    filtered_data[col] = data[data[col].notna()].copy()

    df_var = filtered_data[col]
    
    avg_per_cusip[col] = df_var.groupby('cusip')[col].mean()

    avg_series = avg_per_cusip[col]

    cusip_quintiles[col] = pd.qcut(avg_series, q=5, labels=False) + 1

    quintile_col = f'{col}_quintile'
    
    data[quintile_col] = data['cusip'].map(cusip_quintiles[col])

    missing_by_q = (
        data.groupby(quintile_col)[col]
        .apply(lambda x: x.isna().mean() * 100)
    )
    
    missingness_by_quintile[col] = missing_by_q

fig, ax1 = plt.subplots(figsize=(12, 7))

colors = plt.cm.tab10.colors
axes = [ax1]

var_list = list(missingness_by_quintile.keys())

col0 = var_list[0]
line0, = ax1.plot(
    missingness_by_quintile[col0].index,
    missingness_by_quintile[col0].values,
    label=col0,
    color=colors[0],
    marker='o',
    linestyle='--'
)

lines = [line0]
for i, col in enumerate(var_list[1:], start=1):
    ax = ax1.twinx()
    axes.append(ax)

    ax.spines["right"].set_position(("axes", 1 + 0.1 * (i - 1)))

    line, = ax.plot(
        missingness_by_quintile[col].index,
        missingness_by_quintile[col].values,
        label=col,
        color=colors[i % len(colors)],
        marker='o',
        linestyle='--'
    )
    lines.append(line)

    if i > 1:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)
        for tick in ax.get_yticklines():
            tick.set_visible(False)

ax1.set_xlabel('Firm Quintile Based on Avg Characteristic Value')
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_title('Missingness by Firm Quintile (Each Variable with Own Y-axis)')

labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

plt.tight_layout()
plt.savefig(snakemake.output[10], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
###############################################################################################################
# AUTOCORRELATION IN VARIABLES

variables = ['at', 'ceq', 'csho', 'lct', 'dd1', 'expected_return', 'volatility']  # or your full list

variable_data = {}
mean_autocorr = {}
mean_pvals = {}
results = []
for col in variables:
    df_col = df[['cusip', 'fyear', col]].dropna(subset=[col]).copy()
    corrs = []
    pvals = []

    for _, group in df_col.groupby('cusip'):
        if len(group) < 2:
            continue

        group_sorted = group.sort_values('fyear')
        y = group_sorted[col]
        y_lag = y.shift(1)

        df_valid = pd.DataFrame({'y': y, 'y_lag': y_lag}).dropna()

        if len(df_valid) < 2 or df_valid['y_lag'].std() == 0 or df_valid['y'].std() == 0:
            continue

        df_valid['y_std'] = (df_valid['y'] - df_valid['y'].mean()) / df_valid['y'].std()
        df_valid['y_lag_std'] = (df_valid['y_lag'] - df_valid['y_lag'].mean()) / df_valid['y_lag'].std()

        model = sm.OLS(df_valid['y_std'], df_valid['y_lag_std']).fit()

        beta = model.params['y_lag_std']
        pval = model.pvalues['y_lag_std']

        corrs.append(beta)
        pvals.append(pval)

    mean_autocorr[col] = sum(corrs) / len(corrs) if corrs else None
    mean_pvals[col] = sum(pvals) / len(pvals) if pvals else None

    results.append({
        'variable': col,
        'mean_autocorrelation': mean_autocorr,
        'mean_pvalue': mean_pvals
    })

    print(f"done for {col}")

autocorr_series = pd.Series(mean_autocorr).dropna()

plt.figure(figsize=(12, 6))
plt.plot(autocorr_series.index, autocorr_series.values, marker='o', linestyle='--')
plt.xticks(rotation=45)
plt.ylabel('Mean Lag-1 Autocorrelation')
plt.title('Mean Lag-1 Autocorrelation of Firm-Level Variables')
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[11], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")

###############################################################################################################

# CORRELATION HEATMAP FULL PANEL

numeric_df = df.select_dtypes(include='number')

# Step 2: Compute correlation matrix
corr_matrix = numeric_df.corr()

# Step 3: Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.savefig(snakemake.output[12], dpi=300)
print("Heatmap saved as 'correlation_heatmap.png'")

###############################################################################################################

# DEFAULT SUMMARY TABLE

df['default_status_lag'] = df.groupby('cusip')['default_status'].shift(1)
df['is_new_default'] = (df['default_status'] == 'D') & (df['default_status_lag'].isna())

summary = df.groupby('fyear').agg(
    number_of_observations=('cusip', 'count'),
    number_of_new_defaults=('is_new_default', 'sum')  # sum of True values == count
).reset_index()

summary['proportion_of_defaults'] = summary['number_of_new_defaults'] / summary['number_of_observations']

summary.to_csv(snakemake.output[13])
###############################################################################################################

# DEFAULT VS NON DEFAULT NULL VALUES PLOT OVER THE YEARS

cusips_with_d = df[df['default_status'] == 'D']['cusip'].unique().tolist()
cusips_without_d = df[~df['cusip'].isin(cusips_with_d)]['cusip'].unique().tolist()

df_default = df[df["cusip"].isin(cusips_with_d)]
df_nondefault = df[df["cusip"].isin(cusips_without_d)]

meta_cols = ['fyear', 'cusip', 'default_status']
data_cols = [col for col in df_default.columns if col not in meta_cols]


df_default['null_proportion'] = df_default[data_cols].isnull().sum(axis=1) / len(data_cols)


mean_null_by_year = df_default.groupby('fyear')['null_proportion'].mean().reset_index()

mean_null_by_year['null_proportion'] = mean_null_by_year['null_proportion'].round(4)

df_nondefault['null_proportion'] = df_nondefault[data_cols].isnull().sum(axis=1) / len(data_cols)

mean_null_by_year_nondefault = df_nondefault.groupby('fyear')['null_proportion'].mean().reset_index()

mean_null_by_year_nondefault['null_proportion'] = mean_null_by_year_nondefault['null_proportion'].round(4)

merged_null_proportions = mean_null_by_year.merge(
    mean_null_by_year_nondefault,
    on='fyear',
    how='outer',
    suffixes=('_default', '_nondefault')
)

merged_null_proportions = merged_null_proportions.sort_values('fyear').reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(
    merged_null_proportions['fyear'], 
    merged_null_proportions['null_proportion_default'], 
    label='Defaulting Firms', 
    marker='o'
)
plt.plot(
    merged_null_proportions['fyear'], 
    merged_null_proportions['null_proportion_nondefault'], 
    label='Non-defaulting Firms', 
    marker='o'
)

plt.title('Mean Proportion of Null Values by Year')
plt.xlabel('Year')
plt.ylabel('Mean Null Proportion')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(snakemake.output[14], dpi=300)
###############################################################################################################