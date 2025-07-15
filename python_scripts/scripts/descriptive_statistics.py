import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Data/Prepared Data/prepared_data_merton_pd.csv")

# Plotting PD for each quality rating: Provides confidence in PD estimation - Idea: A+(has minimum), LIQ(has max), D(has second max)

mean_pd = df.groupby('spcsrc')['PD'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10,5))
mean_pd.plot(kind = 'bar')
plt.xlabel('SnP Quality Rating')
plt.ylabel('Mean PD')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results_quarterly/plots/PD by quality rating.png', dpi=300)


# PD by year and quarter

pd_by_yq = df.groupby(['fyearq', 'fqtr'])['PD'].mean().reset_index()

pd_by_yq['year_qtr'] = pd_by_yq['fyearq'].astype(str) + ' Q' + pd_by_yq['fqtr'].astype(str)

plt.figure(figsize=(16,6))
plt.plot(pd_by_yq['year_qtr'], pd_by_yq['PD'], marker='o')
plt.xlabel('Calender Year and Quarter')
plt.ylabel('Mean PD')

xtick_indices = list(range(0, len(pd_by_yq), 4))
plt.xticks(xtick_indices, pd_by_yq['year_qtr'].iloc[xtick_indices], rotation=45)

plt.tight_layout()
plt.savefig('results_quarterly/plots/PD with time.png', dpi=300)


# Asset vol and stock vol co movement

pd_by_yq = df.groupby(['fyearq', 'fqtr'])[['volatility', 'sigma_A']].mean().reset_index()
pd_by_yq['year_qtr'] = pd_by_yq['fyearq'].astype(str) + ' Q' + pd_by_yq['fqtr'].astype(str)

plt.figure(figsize=(16,6))

plt.plot(pd_by_yq['year_qtr'], pd_by_yq['volatility'], marker='o', color='blue', label='Stock volatility')
plt.plot(pd_by_yq['year_qtr'], pd_by_yq['sigma_A'], marker='o', color='red', label='Asset volatility')

plt.xlabel('Calendar Year and Quarter')
plt.ylabel('Volatility')
xtick_indices = list(range(0, len(pd_by_yq), 4))
plt.xticks(xtick_indices, pd_by_yq['year_qtr'].iloc[xtick_indices], rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('results_quarterly/plots/Eq and Asset sigma with time.png', dpi=300)


# Number of defaulting firms per year

df_unique_cusips = (
    df[df['default'] == 1]
    .groupby('fyearq')['cusip']
    .nunique()
    .reset_index()
    .rename(columns={'cusip': 'unique_cusips_with_default'})
)

df_unique_cusips.to_csv("results_quarterly/tables/number_of_defaults_per_year.csv", index = False)

# Plotting the quarterly total net negative news about biodiversity

df1 = pd.read_csv("Data/Raw Data/nyt_indices.csv")

df1['date'] = pd.to_datetime(df1['date'], format='%d%b%Y')
df1 = df1[(df1['date'].dt.year >= 2006) & (df1['date'].dt.year <= 2024)]
df1['year'] = df1['date'].dt.year
df1['quarter'] = df1['date'].dt.quarter
df_quarterly = df1.groupby(['year', 'quarter'])['biodiversity'].sum().reset_index()
df_quarterly['year_quarter'] = df_quarterly['year'].astype(str) + ' Q' + df_quarterly['quarter'].astype(str)

plt.figure(figsize=(10, 5))
plt.plot(df_quarterly['year_quarter'], df_quarterly['biodiversity'], marker='o')
plt.xlabel('Year-Quarter')
plt.ylabel('Sum of net negative biodiversity news')
plt.xticks(
    ticks=range(0, len(df_quarterly['year_quarter']), 4),
    labels=df_quarterly['year_quarter'][::4],
    rotation=45
)
plt.tight_layout()
plt.savefig('results_quarterly/plots/quarterly total negative biod news.png', dpi=300)


# same figure with event - Trump makes changes to the ESA

plt.figure(figsize=(10, 5))
plt.plot(df_quarterly['year_quarter'], df_quarterly['biodiversity'], marker='o')
plt.xlabel('Year-Quarter')
plt.ylabel('Sum of net negative biodiversity news')
plt.xticks(
    ticks=range(0, len(df_quarterly['year_quarter']), 4),
    labels=df_quarterly['year_quarter'][::4],
    rotation=45
)
plt.tight_layout()

# Mark the 3rd quarter of 2019 with a vertical red dotted line
q3_2019_index = df_quarterly[
    (df_quarterly['year'] == 2019) & (df_quarterly['quarter'] == 3)
].index[0]
plt.axvline(x=q3_2019_index, color='red', linestyle=':', label='August 12, Trump announces changes to ESA')

plt.legend()
plt.savefig('results_quarterly/plots/quarterly total negative biod news with event.png', dpi=300)


df_clipped = df_final[df_final['fyearq'] >= 2018]

df_clipped["average_risk"].nunique()

df_clipped['rank'] = df_clipped['average_risk'].rank(method='dense', ascending=False)

gics4_avg_rank = df_clipped.groupby('gics4')['rank'].mean().to_dict()
print(gics4_avg_rank)

sorted_gics4_avg_rank = dict(sorted(gics4_avg_rank.items(), key=lambda item: item[1], reverse=True))
df_clipped.columns

# Filter for gics4 == 3020
df_3020 = df_clipped[df_clipped['gics4'] == "5020"]
df_3020['op_margin'] = df_3020['xoprq'] / df_3020['saleq']

# Group by fyearq and fqtr, then calculate mean PD
mean_pd = df_3020.groupby(['fyearq', 'fqtr'])['op_margin'].mean().reset_index()

# Create a 'date' column for better plotting (optional)
mean_pd['date'] = mean_pd['fyearq'].astype(str) + 'Q' + mean_pd['fqtr'].astype(str)

# Plot
plt.figure(figsize=(10,5))
plt.plot(mean_pd['date'], mean_pd['op_margin'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('Year-Quarter')
plt.ylabel('Mean PD')
plt.title('Mean PD of gics4 3020 over Time')
plt.tight_layout()
plt.show()



df_sub = df_clipped[df_clipped['gics4'].isin(['3020', '5020'])]

df_sub['op_margin'] = df_sub['xoprq'] / df_sub['saleq']

mask = (df_sub['fyearq'] > 2021) | ((df_sub['fyearq'] == 2021) & (df_sub['fqtr'] <= 1))
df_plot = df_sub.loc[mask].copy()
df_plot['period'] = pd.PeriodIndex(year=df_plot['fyearq'], quarter=df_plot['fqtr'], freq='Q')
mean_op = (
    df_plot
    .groupby(['period', 'gics4'])['op_margin']
    .mean()
    .reset_index()
    .pivot(index='period', columns='gics4', values='op_margin')
)

# 6) Plot
plt.figure(figsize=(10, 5))
for industry in mean_op.columns:
    plt.plot(mean_op.index.to_timestamp(), mean_op[industry], marker='o', label=industry)

plt.xlabel('Quarter')
plt.ylabel('Mean Operating Margin')
plt.title('Mean Operating Margin for GICS4 3020 vs 5020 (Up to 2020 Q2)')
plt.legend(title='GICS4')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()