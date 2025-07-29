import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(snakemake.input[0])

# Plotting PD for each quality rating: Provides confidence in PD estimation - Idea: A+(has minimum), LIQ(has max), D(has second max)

mean_pd = df.groupby('spcsrc')['PD'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10,5))
mean_pd.plot(kind = 'bar')
plt.xlabel('SnP Quality Rating')
plt.ylabel('Mean PD')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(snakemake.output[0], dpi=300)


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
plt.savefig(snakemake.output[1], dpi=300)


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
plt.savefig(snakemake.output[2], dpi=300)


# Number of defaulting firms per year

df_unique_cusips = (
    df[df['default'] == 1]
    .groupby('fyearq')['cusip']
    .nunique()
    .reset_index()
    .rename(columns={'cusip': 'unique_cusips_with_default'})
)

df_unique_cusips.to_csv(snakemake.output[3], index = False)

# Plotting the quarterly total net negative news about biodiversity

df1 = pd.read_csv(snakemake.input[1])

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
plt.savefig(snakemake.output[4], dpi=300)

