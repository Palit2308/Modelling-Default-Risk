import pandas as pd
import numpy as np
from fancyimpute import IterativeSVD
##################################################################################################################

def iterative_svd_impute(df):

    df = df.set_index(["cusip", "fyear"]).sort_index()
    data = df.copy()

    mean = df.mean()
    std = df.std()
    stz_df = (df - mean) / std

    nan_mask = data.isna()

    imputed_list = []

    for t in stz_df.index.get_level_values('fyear').unique():
        df_t = stz_df.xs(t, level='fyear')  # df_t has index = cusip
        imputed = IterativeSVD().fit_transform(df_t)
        imputed_df = pd.DataFrame(imputed, index=df_t.index, columns=df_t.columns)
        
        imputed_df['fyear'] = t
        imputed_df['cusip'] = imputed_df.index  # preserve original index
        
        imputed_df = imputed_df.set_index(['cusip', 'fyear']).sort_index()
        imputed_list.append(imputed_df)

    # Combine all
    stz_df_imputed = pd.concat(imputed_list)

    # Copy original data for final imputation
    final_imputed_df = data.copy()

    for col in data.columns:
        missing_idx = data.index[nan_mask[col]]
        for idx in missing_idx:
            final_imputed_df.loc[idx, col] = stz_df_imputed.loc[idx, col] * std[col] + mean[col]
            print(f"done for {idx}")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    final_imputed_df = final_imputed_df.reset_index()

    return final_imputed_df

##################################################################################################################