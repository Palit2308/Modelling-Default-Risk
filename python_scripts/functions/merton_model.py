import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

# Merton Model Functions

def merton_solver(E, sigma_E, D, r, T=1.0):
    """
    Solves for asset value A and asset volatility sigma_A using the Merton model equations.

    Parameters:
    - E: Market value of equity
    - sigma_E: Volatility of equity
    - D: Face value of debt
    - r: Risk-free rate
    - T: Time to maturity (default 1 year)

    Returns:
    - A: Estimated asset value
    - sigma_A: Estimated asset volatility
    """

    def equations(vars):
        A, sigma_A = vars
        d = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
        eq1 = E - (A * norm.cdf(d) - D * np.exp(-r * T) * norm.cdf(d - sigma_A * np.sqrt(T)))
        eq2 = sigma_E - (A / E) * norm.cdf(d) * sigma_A
        return [eq1, eq2]

    # Initial guesses
    A0 = E + D
    sigma_A0 = sigma_E * E / (E + D)
    solution = root(equations, [A0, sigma_A0])
    if solution.success:
        return solution.x
    else:
        return [np.nan, np.nan]

def distance_to_default(A, D, sigma_A, mu_A, T=1.0):
    """
    Computes Distance to Default (DD) using the Merton model output.

    Parameters:
    - A: Asset value
    - D: Face value of debt
    - sigma_A: Asset volatility
    - mu_A: Expected return on assets
    - T: Time to maturity

    Returns:
    - DD: Distance to Default
    """

    if D is None or D == 0 or A is None or A <= 0:
        return np.nan
    
    numerator = np.log(A / D) + (mu_A - 0.5 * sigma_A**2) * T
    denominator = sigma_A * np.sqrt(T)
    if denominator == 0:
            return np.nan
    return numerator / denominator
    

def two_system_merton(df, E, sigma_E, D, rf, mu_a):

    df_merton = df.copy()

    df_merton.set_index(["cusip", "fyearq", "fqtr"], inplace=True)

    df_merton[['A', 'sigma_A']] = df_merton.apply(
    lambda row: pd.Series(merton_solver(row[E], row[sigma_E], row[D], row[rf], T=1.0)),
    axis=1
    )
    print("step1 done")
    df_merton['DD'] = df_merton.apply(
        lambda row: distance_to_default(row['A'], row[D], row['sigma_A'], row[mu_a], T=1.0),
        axis=1
    )
    print("step2 done")
    df_merton['PD'] = norm.cdf(-df_merton['DD'])

    return df_merton