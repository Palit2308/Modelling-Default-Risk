import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
#############################################################################################################################

def propensity_matching_summary_table(df):

    df = df.reset_index()

    defaulting_cusips = set(df[df["default"] == 1]["cusip"].unique())

    non_default_df = df[~df["cusip"].isin(defaulting_cusips)]

    control_candidates = []

    for cusip, group in non_default_df.groupby("cusip"):
        years = sorted(group["fyear"].unique())
        for i in range(len(years) - 5):
            window = years[i:i + 6]
            if window[-1] - window[0] == 5:
                control_candidates.append((cusip, window[-1]))  # last year is pseudo-default year
                break  # stop at first valid 6-year window

    df_qualified_control = pd.DataFrame(control_candidates, columns=["cusip", "pseudo_default_year"])

    default_qualified_cusips = []

    for cusip, group in df.groupby('cusip'):
        default_years = group[group['default'] == 1]['fyear']
        
        for year in default_years:
            required_years = set([year - i for i in range(6)])
            available_years = set(group['fyear'])

            if required_years.issubset(available_years):
                default_qualified_cusips.append((cusip, year))
                break 


    df_qualified_default = pd.DataFrame(default_qualified_cusips, columns=["cusip", "default_year"])


    df_qualified_default['treatment'] = 1
    df_qualified_control['treatment'] = 0
    df_qualified_control = df_qualified_control.rename(columns={'pseudo_default_year': 'default_year'})

    df_matched_pool = pd.concat([df_qualified_default, df_qualified_control], ignore_index=True)

    data = df.copy()

    data['debt_level_0'] = data['lct'] + data['dd1']

    data['A/D_0_rf'] = data['A_0_rf'] / data['debt_level_0']

    data['BA/D_0_rf'] = data['at'] / data['debt_level_0']

    data['equity_return'] = data["expected_return"] * 252

    data['annualised_volatiltiy'] = data["volatility"] * np.sqrt(252)

    features = ['at', 'ch', 'ceq', 'csho', 'lct', 'dd1', 'dltt', 'dt', 'gp', 'ebit', 'sale', 'prcc_c', 
                'debt_level_0', 'A/D_0_rf', 'BA/D_0_rf', 'equity_return', 'annualised_volatiltiy' ]

    df_features = data.groupby('cusip')[features].mean().reset_index()

    df_matched_pool = df_matched_pool.merge(df_features, on='cusip')

    X = df_matched_pool[features]
    y = df_matched_pool['treatment']

    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
    y_clean = y.loc[X_clean.index]  

    model = LogisticRegression()
    model.fit(X_clean, y_clean)

    df_matched_pool = df_matched_pool.loc[X_clean.index]

    df_matched_pool['propensity_score'] = model.predict_proba(X_clean)[:, 1] 

    treated = df_matched_pool[df_matched_pool['treatment'] == 1]
    control = df_matched_pool[df_matched_pool['treatment'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])

    distances, indices = nn.kneighbors(treated[['propensity_score']])
    control_indices = indices.flatten()

    matched = pd.DataFrame({
        'default_cusip': treated['cusip'].values,
        'default_year': treated['default_year'].values,
        'control_cusip': control.iloc[control_indices]['cusip'].values,
        'control_year': control.iloc[control_indices]['default_year'].values,
        'distance': distances.flatten(),
        'pscore_default': treated['propensity_score'].values,
        'pscore_control': control.iloc[control_indices]['propensity_score'].values
    })

    matched_controls = matched[["control_cusip", "control_year"]].rename(
        columns={"control_cusip": "cusip", "control_year": "default_year"}
    )
    matched_defaults = matched[["default_cusip", "default_year"]].rename(
        columns={"default_cusip": "cusip", "default_year": "default_year"}
    )

    filtered_df_default = pd.merge(
        df.reset_index(),
        matched_defaults,
        on="cusip"
    )

    filtered_df_default = filtered_df_default[
        (filtered_df_default['fyear'] >= filtered_df_default['default_year'] - 5) &
        (filtered_df_default['fyear'] <= filtered_df_default['default_year'])
    ]

    filtered_df_default = filtered_df_default.set_index(["cusip", "fyear"]).sort_index()

    filtered_df_default['debt_level_0'] = filtered_df_default['lct'] + filtered_df_default['dd1']

    filtered_df_default['A/D_0_rf'] = filtered_df_default['A_0_rf'] / filtered_df_default['debt_level_0']

    filtered_df_default['BA/D_0_rf'] = filtered_df_default['at'] / filtered_df_default['debt_level_0']
    
    filtered_df_default['A/LTD_0_rf'] = filtered_df_default['A_0_rf'] / filtered_df_default['dltt']

    filtered_df_default['BA/LTD'] = filtered_df_default['at'] / filtered_df_default['dltt']

    filtered_df_default['equity_return'] = filtered_df_default["expected_return"] * 252

    filtered_df_default['annualised_volatiltiy'] = filtered_df_default["volatility"] * np.sqrt(252)

    filtered_df_default = filtered_df_default.reset_index()
    filtered_df_default["years_to_default"] = filtered_df_default["fyear"] - filtered_df_default["default_year"]

    filtered_df_default = filtered_df_default.replace([np.inf, -np.inf], np.nan)

    summary_default = filtered_df_default.groupby("years_to_default").agg({
        "PD_0_rf": "mean",
        "DD_0_rf": "mean",
        "A/D_0_rf": "mean",
        "equity_return": "mean",
        "annualised_volatiltiy": "median",
        "sigma_A_0_rf" : "median",
        "BA/D_0_rf" : "mean",
        'A/LTD_0_rf' : "mean",
        "BA/LTD" : "mean"
    }).reset_index()


    filtered_df_controls = pd.merge(
        df.reset_index(),
        matched_controls,
        on="cusip"
    )

    filtered_df_controls = filtered_df_controls[
        (filtered_df_controls['fyear'] >= filtered_df_controls['default_year'] - 5) &
        (filtered_df_controls['fyear'] <= filtered_df_controls['default_year'])
    ]

    filtered_df_controls = filtered_df_controls.set_index(["cusip", "fyear"]).sort_index()

    filtered_df_controls['debt_level_0'] = filtered_df_controls['lct'] + filtered_df_controls['dd1']

    filtered_df_controls['A/D_0_rf'] = filtered_df_controls['A_0_rf'] / filtered_df_controls['debt_level_0']

    filtered_df_controls['BA/D_0_rf'] = filtered_df_controls['at'] / filtered_df_controls['debt_level_0']

    filtered_df_controls['A/LTD_0_rf'] = filtered_df_controls['A_0_rf'] / filtered_df_controls['dltt']

    filtered_df_controls['BA/LTD'] = filtered_df_controls['at'] / filtered_df_controls['dltt']

    filtered_df_controls['equity_return'] = filtered_df_controls["expected_return"] * 252

    filtered_df_controls['annualised_volatiltiy'] = filtered_df_controls["volatility"] * np.sqrt(252)

    filtered_df_controls = filtered_df_controls.reset_index()
    filtered_df_controls["years_to_default"] = filtered_df_controls["fyear"] - filtered_df_controls["default_year"]

    filtered_df_controls = filtered_df_controls.replace([np.inf, -np.inf], np.nan)

    summary_controls = filtered_df_controls.groupby("years_to_default").agg({
        "PD_0_rf": "mean",
        "DD_0_rf": "mean",
        "A/D_0_rf": "mean",
        "equity_return": "median",
        "annualised_volatiltiy": "median",
        "sigma_A_0_rf" : "median",
        "BA/D_0_rf" : "mean",
        'A/LTD_0_rf' : "mean",
        "BA/LTD" : "mean"
    }).reset_index()

    return summary_default, summary_controls