import pandas as pd
import numpy as np
from fancyimpute import IterativeSVD

# Function for imputation

def iterative_svd_impute(df):
    
    df = df.set_index(["cusip", "fyearq", "fqtr"]).sort_index()
    data = df.copy()

    num_cols = data.select_dtypes(include=[np.number]).columns
    data_num = data[num_cols]

    mean = data_num.mean()
    std = data_num.std()
    stz_df = (data_num - mean) / std

    nan_mask = data.isna()

    imputed_list = []

    for t in stz_df.index.get_level_values('fqtr').unique():
        df_t = stz_df.xs(t, level='fqtr')  # df_t has MultiIndex (cusip, fyearq)
        # Save a DataFrame of imputed values
        imputed = IterativeSVD().fit_transform(df_t)
        imputed_df = pd.DataFrame(imputed, index=df_t.index, columns=df_t.columns)
        # Add 'fqtr' as a column to preserve for reindexing
        imputed_df['fqtr'] = t
        imputed_list.append(imputed_df)

    # Combine all quarters
    stz_df_imputed = pd.concat(imputed_list)
    # Now, stz_df_imputed index = (cusip, fyearq), 'fqtr' is a column
    # Reset and set correct multiindex
    stz_df_imputed = stz_df_imputed.reset_index().set_index(['cusip', 'fyearq', 'fqtr']).sort_index()
    # Remove any non-feature columns
    stz_df_imputed = stz_df_imputed[data.columns]  # just features

    # Inverse transform to original scale, only at missing positions
    final_imputed_df = data.copy()
    for col in data.columns:
        # Only fill NaNs
        mask = nan_mask[col]
        # Use stz_df_imputed to impute, undo standardization
        final_imputed_df.loc[mask, col] = (stz_df_imputed.loc[mask, col] * std[col]) + mean[col]
        # print(f"Imputed {col}")

    final_imputed_df = final_imputed_df.reset_index()
    return final_imputed_df