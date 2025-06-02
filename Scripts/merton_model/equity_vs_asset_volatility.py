import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


input1 = snakemake.input[0]
input2 = snakemake.input[1]

df = pd.read_csv(input1)
df1 = pd.read_csv(input2)

df["annualised_volatility"] = df["volatility"] * np.sqrt(252)
df_plot = df.groupby('fyear').agg({
    'annualised_volatility': 'mean',
    'sigma_A_0_rf': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))

# Plot both series on the same y-axis
plt.plot(df_plot['fyear'], df_plot['annualised_volatility'], 
         color='tab:blue', linestyle='-', linewidth=2, label='Equity Volatility (Annualized)')
plt.plot(df_plot['fyear'], df_plot['sigma_A_0_rf'], 
         color='tab:red', linestyle='-', linewidth=2, label='Asset Volatility (σ_A)')

plt.xticks(df_plot['fyear'], rotation=45)  # Force all years + rotate labels
plt.xlabel('Year', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.title('Equity vs. Asset Volatility by Year', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=1)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(snakemake.output[0], dpi=300)

################################################################################################################################

df1["annualised_volatility"] = df1["volatility"] * np.sqrt(252)
df_plot1 = df1.groupby('fyear').agg({
    'annualised_volatility': 'mean',
    'sigma_A_0_rf': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))

# Plot both series on the same y-axis
plt.plot(df_plot1['fyear'], df_plot1['annualised_volatility'], 
         color='tab:blue', linestyle='-', linewidth=2, label='Equity Volatility (Annualized)')
plt.plot(df_plot1['fyear'], df_plot1['sigma_A_0_rf'], 
         color='tab:red', linestyle='-', linewidth=2, label='Asset Volatility (σ_A)')

plt.xticks(df_plot1['fyear'], rotation=45)  # Force all years + rotate labels
plt.xlabel('Year', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.title('Equity vs. Asset Volatility by Year', fontsize=14, pad=20)
plt.legend(fontsize=10, framealpha=1)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(snakemake.output[1], dpi=300)