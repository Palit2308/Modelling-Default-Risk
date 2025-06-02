import pandas as pd
import matplotlib.pyplot as plt
from Scripts.functions.handling_prepared_data.create_default_indicator import create_default_indicator
from Scripts.functions.handling_prepared_data.joining_stocks_firms_data import join_stocks_firms_data
import numpy as np
import seaborn as sns
from Scripts.functions.handling_prepared_data.iterative_svd_impute import iterative_svd_impute
############################################################################################################

input1 = snakemake.input[0]       
input2 = snakemake.input[1]
input3 = snakemake.input[2]

df_annual = pd.read_csv(input1)
df_stocks = pd.read_csv(input2)
df_interest = pd.read_excel(input3)

df = join_stocks_firms_data(df_stocks, df_annual, df_interest)
df = create_default_indicator(df)

imputed_df = iterative_svd_impute(df)

imputed_df.to_csv(snakemake.output[0], index = False)