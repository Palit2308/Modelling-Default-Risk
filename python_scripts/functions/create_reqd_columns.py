import pandas as pd
import numpy as np

# Function to create the required columns for merton modelling

def create_reqd_colums(data, year = "fyearq", ret = "returns", dltt = "dlttq", k = 0.5):

    df = data.copy()
    df = df[df[year] <= 2024]
    df["eqrt"] = df[ret] * 63 # Making average daily log returns into cumulative quarterly log returns 
    df["volatility"] = df["volatility"] * 2 # Annualising volatility
    df["short_term_debt"] = df["lctq"] + df["dd1q"]
    df["long_term_debt"] = df[dltt]
    df["debt_level"] = df["short_term_debt"] + k * df["long_term_debt"]
    df['mu_a_max'] = df[['rf', 'eqrt']].max(axis=1)

    df = df.drop(columns=[ret, dltt])
    return df