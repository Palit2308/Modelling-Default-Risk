import pandas as pd
import matplotlib.pyplot as plt
from Scripts.functions.handling_prepared_data.create_default_indicator import create_default_indicator
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
df = create_default_indicator(df)

###############################################################################################################

total_rows = len(df)
complete_rows = df.dropna().shape[0]
incomplete_rows = total_rows - complete_rows

pct_complete = (complete_rows / total_rows) * 100
pct_incomplete = (incomplete_rows / total_rows) * 100

###############################################################################################################

df_complete = df.dropna()

rows_with_default_1_complete = (df_complete["default"] == 1).sum()
rows_with_default_0_complete = (df_complete["default"] == 0).sum()

number_of_defaulting_cusips_with_full_data = df_complete[df_complete['default'] == 1]['cusip'].nunique()

grouped = df_complete.groupby('cusip')
non_defaulting_cusips = [cusip for cusip, group in grouped if (group['default'] == 0).all()]
number_of_non_defaulting_cusips_with_full_data = len(non_defaulting_cusips)

###############################################################################################################

cusip_summary = []

for cusip, group in df.groupby('cusip'):
    start_year = group['fyear'].min()
    end_year = group['fyear'].max()
    has_defaulted = (group['default'] == 1).any()

    total_cells = group.shape[0] * (group.shape[1] - 2)
    num_nulls = group.drop(columns=['cusip', 'fyear']).isna().sum().sum()
    pct_null = (num_nulls / total_cells) * 100 if total_cells > 0 else np.nan

    cusip_summary.append({
        'cusip': cusip,
        'start_year': start_year,
        'end_year': end_year,
        'has_defaulted': has_defaulted,
        'pct_null_values': pct_null
    })

cusip_panel_summary = pd.DataFrame(cusip_summary)

cusip_panel_summary.to_csv(snakemake.output[0], index = False)

plt.figure(figsize=(10, 6))
plt.hist(cusip_panel_summary['pct_null_values'], bins=30, edgecolor='black')
plt.xlabel('% of Null Values in CUSIP Panel')
plt.ylabel('Number of CUSIPs')
plt.title('Distribution of Null Value Percentage Across CUSIPs')
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[1], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")

###############################################################################################################

plt.figure(figsize=(10, 6))

sns.kdeplot(
    data=cusip_panel_summary[cusip_panel_summary['has_defaulted'] == True],
    x='pct_null_values',
    fill=True,
    label='Defaulted',
    alpha=0.6
)

sns.kdeplot(
    data=cusip_panel_summary[cusip_panel_summary['has_defaulted'] == False],
    x='pct_null_values',
    fill=True,
    label='Non-defaulted',
    alpha=0.6
)

plt.xlabel('% of Null Values in CUSIP Panel')
plt.ylabel('Density')
plt.title('Distribution of Missingness by Default Status')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[2], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")

###############################################################################################################

low_missing = cusip_panel_summary[cusip_panel_summary['pct_null_values'] < 40]
high_missing = cusip_panel_summary[cusip_panel_summary['pct_null_values'] >= 40]

low_defaulted = low_missing[low_missing['has_defaulted'] == True].shape[0]
low_nondefaulted = low_missing[low_missing['has_defaulted'] == False].shape[0]

high_defaulted = high_missing[high_missing['has_defaulted'] == True].shape[0]
high_nondefaulted = high_missing[high_missing['has_defaulted'] == False].shape[0]

summary_df = pd.DataFrame({
    'defaulted': [low_defaulted, high_defaulted],
    'non_defaulted': [low_nondefaulted, high_nondefaulted]
}, index=['< 40% nulls', '>= 40% nulls'])

summary_df.to_csv(snakemake.output[3], index = False)

###############################################################################################################