import pandas as pd


def join_stocks_firms_data(df_stocks, df_annual, df_interest):

    df_stocks = df_stocks.rename(columns={'year': 'fyear'})

    df_interest = df_interest.rename(columns={'Year': 'fyear'})

    df_joined = df_annual.merge(df_stocks, on=['cusip', 'fyear'], how='left')

    df_joined = df_joined.sort_values(by=["cusip", "fyear"])

    df_joined = df_joined.drop(columns="Unnamed: 0")

    df_joined = df_joined.merge(df_interest[['fyear', 'Average Closing Price']], on='fyear', how='left')

    df_joined = df_joined.rename(columns={"Average Closing Price": "rf"})

    return df_joined