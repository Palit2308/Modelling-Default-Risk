import pandas as pd
import matplotlib.pyplot as plt
from Scripts.functions.handling_prepared_data.create_default_indicator import create_default_indicator
from Scripts.functions.handling_prepared_data.joining_stocks_firms_data import join_stocks_firms_data
from Scripts.functions.merton_model.two_eqn_merton import two_system_merton
from Scripts.functions.merton_model.single_eqn_merton import single_system_merton
from Scripts.functions.merton_model.auc_pauc_table import compute_auc_pauc
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import root
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from numpy import trapz
from MLstatkit.stats import Delong_test
###############################################################################################################
input1 = snakemake.input[0]

df = pd.read_csv("Data/prepared_datasets/financials_annual_merton_imputed.csv")

#########################################################################################################################

k_val = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mua_val = ["mu_a_max", "rf", "equity_return"]

k_list = []
mua_list = []
auc_list = []
pauc_list = []
p_value_list = []
df_roc_two_system = df.set_index(["cusip", "fyear"])

for k in k_val:
    for mua in mua_val: 
        df_merton = two_system_merton(df = df, k = k, E = 'mktval', sigma_E= 'annualised_volatility', D = 'debt_level', rf = 'rf', mu_a = mua)
        df_merton = df_merton.rename(columns = {"PD" : f"PD_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"A" : f"A_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"sigma_A" : f"sigma_A_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"DD" : f"DD_{k}_{mua}"})
        df_roc_two_system = df_roc_two_system.merge(df_merton[[f"PD_{k}_{mua}", f"DD_{k}_{mua}", f"A_{k}_{mua}", f"sigma_A_{k}_{mua}"]], left_index=True, right_index=True, how='left')
        print(f"done two system for {k} - {mua}")

df_roc_two_system.reset_index().to_csv(snakemake.output[0], index = False)
df_roc_two_system = df_roc_two_system.dropna()

for k in k_val:
    for mua in mua_val:
        auc, pauc = compute_auc_pauc(df_roc_two_system['default'], df_roc_two_system[f'PD_{k}_{mua}'])
        p_value = Delong_test(df_roc_two_system["default"], df_roc_two_system[f'PD_{k}_{mua}'], df_roc_two_system[f"PD_0.5_{mua}"])[1]
        k_list.append(k)
        mua_list.append(mua)
        auc_list.append(auc)
        pauc_list.append(pauc)
        p_value_list.append(p_value)

results_table = pd.DataFrame({
    "k": k_list,
    "mu_a": mua_list,
    "AUC": auc_list,
    "pAUC": pauc_list,
    "AUC Difference P Value" : p_value_list
})

results_table.to_csv(snakemake.output[1], index = False)

#########################################################################################################################

k_val = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mua_val = ["mu_a_max", "rf", "mu_a"]

k_list = []
mua_list = []
auc_list = []
pauc_list = []
p_value_list = []
df_roc = df.set_index(["cusip", "fyear"])

for k in k_val:
    for mua in mua_val: 
        df_merton = single_system_merton(df = df, k = k, A = "market_assets", sigma_A = "vol_naive", D = "debt_level", rf = "rf", mu_a = mua)
        df_merton = df_merton.rename(columns = {"PD" : f"PD_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"DD" : f"DD_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"market_assets" : f"A_{k}_{mua}"})
        df_merton = df_merton.rename(columns = {"vol_naive" : f"sigma_A_{k}_{mua}"})
        df_roc = df_roc.merge(df_merton[[f"PD_{k}_{mua}", f"DD_{k}_{mua}", f"A_{k}_{mua}", f"sigma_A_{k}_{mua}"]], left_index=True, right_index=True, how='left')
        print(f"done one system for {k} - {mua}")

df_roc.reset_index().to_csv(snakemake.output[2], index = False)
df_roc = df_roc.dropna()

for k in k_val:
    for mua in mua_val:
        auc, pauc = compute_auc_pauc(df_roc['default'], df_roc[f'PD_{k}_{mua}'])
        p_value = Delong_test(df_roc["default"], df_roc[f'PD_{k}_{mua}'], df_roc[f"PD_0.5_{mua}"])[1]
        k_list.append(k)
        mua_list.append(mua)
        auc_list.append(auc)
        pauc_list.append(pauc)
        p_value_list.append(p_value)

results_table_single = pd.DataFrame({
    "k": k_list,
    "mu_a": mua_list,
    "AUC": auc_list,
    "pAUC": pauc_list,
    "AUC Difference P Value" : p_value_list
})

results_table_single.to_csv(snakemake.output[3], index = False)

#########################################################################################################################