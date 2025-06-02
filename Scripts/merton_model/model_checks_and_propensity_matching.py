import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from Scripts.functions.merton_model.propensity_match_summary_table import propensity_matching_summary_table
#############################################################################################################################

df_two_system = pd.read_csv(snakemake.input[0]).set_index(["cusip", "fyear"])
df_one_system = pd.read_csv(snakemake.input[1]).set_index(["cusip", "fyear"])
#############################################################################################################################

# SUMMARY TABLE FOR 2 SYSTEM MODEL

k_val = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mua_val = ["mu_a_max", "rf", "equity_return"]

# Store results
summary_stats = []

for k in k_val:
    for mua in mua_val:
        col = f'PD_{k}_{mua}'
        if col not in df_two_system.columns:
            print(f"Column missing: {col}")
            continue
        
        df_valid = df_two_system[[col, 'default']].dropna()

        pd_default = df_valid[df_valid['default'] == 1][col]
        pd_non_default = df_valid[df_valid['default'] == 0][col]

        mean_default = pd_default.mean()
        median_default = pd_default.median()
        min_default = pd_default.min()
        max_default = pd_default.max()

        mean_non_default = pd_non_default.mean()
        median_non_default = pd_non_default.median()
        min_non_default = pd_non_default.min()
        max_non_default = pd_non_default.max()

        # t-test
        t_stat, p_val = ttest_ind(pd_default, pd_non_default, equal_var=False, nan_policy='omit', alternative='greater' )

        summary_stats.append({
            "k": k,
            "mu_a": mua,
            "mean_default": mean_default,
            "median_default": median_default,
            "min_default": min_default,
            "max_default": max_default,
            "mean_non_default": mean_non_default,
            "median_non_default": median_non_default,
            "min_non_default": min_non_default,
            "max_non_default": max_non_default,
            "ttest_pval": p_val
        })


df_summary = pd.DataFrame(summary_stats)

df_summary.to_csv(snakemake.output[0], index = False)
#############################################################################################################################

# SUMMARY TABLE FOR ONE SYSTEM MODEL

k_val = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mua_val = ["mu_a_max", "rf", "mu_a"]

# Store results
summary_stats = []

for k in k_val:
    for mua in mua_val:
        col = f'PD_{k}_{mua}'
        if col not in df_one_system.columns:
            print(f"Column missing: {col}")
            continue
        
        df_valid = df_one_system[[col, 'default']].dropna()

        pd_default = df_valid[df_valid['default'] == 1][col]
        pd_non_default = df_valid[df_valid['default'] == 0][col]

        mean_default = pd_default.mean()
        median_default = pd_default.median()
        min_default = pd_default.min()
        max_default = pd_default.max()

        mean_non_default = pd_non_default.mean()
        median_non_default = pd_non_default.median()
        min_non_default = pd_non_default.min()
        max_non_default = pd_non_default.max()

        # t-test
        t_stat, p_val = ttest_ind(pd_default, pd_non_default, equal_var=False, nan_policy='omit', alternative='greater' )

        summary_stats.append({
            "k": k,
            "mu_a": mua,
            "mean_default": mean_default,
            "median_default": median_default,
            "min_default": min_default,
            "max_default": max_default,
            "mean_non_default": mean_non_default,
            "median_non_default": median_non_default,
            "min_non_default": min_non_default,
            "max_non_default": max_non_default,
            "ttest_pval": p_val
        })


df_summary = pd.DataFrame(summary_stats)

df_summary.to_csv(snakemake.output[1], index = False)
#############################################################################################################################

# PROPENSITY MATCHING FOR 2 SYSTEM MODEL AND GENERATING SUMMARY TABLES FOR TREATMENT AND CONTROL GROUP
# GENERATING PLOTS FOR TWO SYSTEM MODEL

summary_default, summary_controls = propensity_matching_summary_table(df_two_system)

plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["PD_0_rf"], marker='s', label="Mean PD Evolution - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["PD_0_rf"], marker='s', label="Mean PD Evolution - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of PD by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean PD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[2], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["equity_return"], marker='s', label="Median Equity Returns - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["equity_return"], marker='s', label="Median Equity Returns - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Equity Return by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Equity Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[3], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["annualised_volatiltiy"], marker='s', label="Median Equity Vol - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["annualised_volatiltiy"], marker='s', label="Median Equity Vol - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Equity Vol by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Equity Vol")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[4], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["sigma_A_0_rf"], marker='s', label="Median Asset Vol (Implied) - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["sigma_A_0_rf"], marker='s', label="Median Asset Vol (Implied)- Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset Vol by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Asset Vol")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[5], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Debt Ratio by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[6], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["BA/D_0_rf"], marker='s', label="Mean Book Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["BA/D_0_rf"], marker='s', label="Mean Book Asset(Implied) to Debt Ratio - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Book Asset to Debt Ratio by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean BA2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[7], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_default["years_to_default"], summary_default["BA/D_0_rf"], marker='s', label="Mean Asset(Book) to Debt Ratio - Default")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Debt Ratio by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[8], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/LTD_0_rf"], marker='s', label="Mean Asset(Implied) to Long Term Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["A/LTD_0_rf"], marker='s', label="Mean Asset(Implied) to Long Term Debt Ratio - Controls")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Long Term Debt Ratio by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2LTD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[9], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["BA/LTD"], marker='s', label="Mean Asset(Book) to Long Term Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["BA/LTD"], marker='s', label="Mean Asset(Book) to Long Term Debt Ratio - Controls")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Long Term Debt Ratio by Two System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean BA2LTD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[10], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
#############################################################################################################################

# PROPENSITY MATCHING FOR 1 SYSTEM MODEL AND GENERATING SUMMARY TABLES FOR TREATMENT AND CONTROL GROUP
# GENERATING PLOTS FOR ONE SYSTEM MODEL

summary_default, summary_controls = propensity_matching_summary_table(df_one_system)

plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["PD_0_rf"], marker='s', label="Mean PD Evolution - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["PD_0_rf"], marker='s', label="Mean PD Evolution - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of PD by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean PD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[11], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["equity_return"], marker='s', label="Median Equity Returns - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["equity_return"], marker='s', label="Median Equity Returns - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Equity Return by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Equity Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[12], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["annualised_volatiltiy"], marker='s', label="Median Equity Vol - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["annualised_volatiltiy"], marker='s', label="Median Equity Vol - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Equity Vol by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Equity Vol")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[13], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["sigma_A_0_rf"], marker='s', label="Median Asset Vol (Implied) - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["sigma_A_0_rf"], marker='s', label="Median Asset Vol (Implied)- Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset Vol by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Median Asset Vol")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[14], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Debt Ratio by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[15], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["BA/D_0_rf"], marker='s', label="Mean Book Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["BA/D_0_rf"], marker='s', label="Mean Book Asset(Implied) to Debt Ratio - Control")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Book Asset to Debt Ratio by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean BA2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[16], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/D_0_rf"], marker='s', label="Mean Asset(Implied) to Debt Ratio - Default")
plt.plot(summary_default["years_to_default"], summary_default["BA/D_0_rf"], marker='s', label="Mean Asset(Book) to Debt Ratio - Default")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Debt Ratio by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2D")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[17], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["A/LTD_0_rf"], marker='s', label="Mean Asset(Implied) to Long Term Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["A/LTD_0_rf"], marker='s', label="Mean Asset(Implied) to Long Term Debt Ratio - Controls")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Long Term Debt Ratio by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean A2LTD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[18], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")


plt.figure(figsize=(10, 6))

plt.plot(summary_default["years_to_default"], summary_default["BA/LTD"], marker='s', label="Mean Asset(Book) to Long Term Debt Ratio - Default")
plt.plot(summary_controls["years_to_default"], summary_controls["BA/LTD"], marker='s', label="Mean Asset(Book) to Long Term Debt Ratio - Controls")

plt.axvline(0, color='red', linestyle='--', label="Default Year")
plt.title("Trajectories of Asset to Long Term Debt Ratio by One System Model Leading Up to Default")
plt.xlabel("Years to Default (0 = default year)")
plt.ylabel("Mean BA2LTD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(snakemake.output[19], dpi=300)
print("Plot saved as 'cusip_default_imbalance.png'")
#############################################################################################################################
