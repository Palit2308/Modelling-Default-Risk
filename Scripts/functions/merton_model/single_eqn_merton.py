import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import root
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from numpy import trapz
from Scripts.functions.merton_model.create_required_columns import create_reqd_columns_single

def distance_to_default_single(A, D, sigma_A, mu_A):
      numerator = np.log(A / D) + (mu_A - 0.5 * sigma_A**2)
      denominator = sigma_A 
      if denominator == 0:
            return np.nan
      return numerator / denominator

def single_system_merton(df, k, A, sigma_A, D, rf, mu_a):
      
    df = create_reqd_columns_single(df, k)

    df_merton = df[["cusip", "fyear", "market_assets", "vol_naive", "debt_level", "rf", "mu_a", "default"]]

    df_merton.set_index(["cusip", "fyear"], inplace=True)

    df_merton['mu_a_max'] = df_merton[['rf', 'mu_a']].max(axis=1)

    df_merton['DD'] = df_merton.apply(
        lambda row: distance_to_default_single(row[A], row[D], row[sigma_A], row[mu_a]),
        axis=1
    )

    df_merton['PD'] = norm.cdf(-df_merton['DD'])

    return df_merton