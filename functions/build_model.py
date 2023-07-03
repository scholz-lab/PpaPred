import numpy as np
import pandas as pd


def addhistory(df, dt_shifted):
    multshift = df.copy()
    for i in dt_shifted:
        p_shift = df.shift(i)
        n_shift = df.shift(-i)
        multshift = pd.concat([multshift, p_shift.add_suffix(f"_pos{i}"), n_shift.add_suffix(f"_neg{i}")], axis=1)
        
    return multshift

def select_features(df, names):
    if isinstance(names, str):
        if names == 'all':
            names = df.columns
    return df.loc[:, names]

