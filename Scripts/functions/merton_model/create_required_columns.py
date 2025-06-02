import pandas as pd
import numpy as np

def create_reqd_columns_merton(df, k = 0.5):
    df["rf"] = df["rf"] / 100
    df["equity_return"] = df["expected_return"] * 252
    df["annualised_volatility"] = df["volatility"] * np.sqrt(252)
    df["short_term_debt"] = df["lct"] + df["dd1"]
    df["long_term_debt"] = df["dltt"]
    df["mktval"] = df["csho"] * df["prcc_c"]
    df["book_assets"] = df["at"]
    df["debt_level"] = df["short_term_debt"] + k * df["long_term_debt"]

    return df

def create_reqd_columns_single(df, k = 0.5):
    df["rf"] = df["rf"] / 100
    df["equity_return"] = df["expected_return"] * 252
    df["annualised_volatility"] = df["volatility"] * np.sqrt(252)
    df["short_term_debt"] = df["lct"] + df["dd1"]
    df["long_term_debt"] = df["dltt"]
    df["mktval"] = df["csho"] * df["prcc_c"]
    df["book_assets"] = df["at"]
    df["debt_level"] = df["short_term_debt"] + k * df["long_term_debt"]
    df["market_assets"] = df["mktval"] + df["debt_level"]
    df["mu_a"] = df["equity_return"]
    df["vol_naive"] = ((df["mktval"] / df["market_assets"]) * df["annualised_volatility"]) + ((df["debt_level"] / df["market_assets"]) * (0.05 + 0.25 * df["annualised_volatility"]))

    return df