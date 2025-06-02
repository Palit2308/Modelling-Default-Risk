import pandas as pd
import numpy as np

def create_default_indicator(df):
    df["default"] = np.zeros
    df["default"] = (df["default_status"] == 'D').astype(int)

    df = df.drop(columns="default_status")

    return df