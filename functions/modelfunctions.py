# %%
# imports
import pandas as pd

# ML imports
from sklearn.preprocessing import power_transform
from sklearn.base import BaseEstimator, TransformerMixin

# %%
# functions
def add_power_transform(df, cols):
    df.loc[:,cols] = power_transform(df.loc[:,cols])
    return df.add_suffix("_tr")  #pd.concat([df, pd.DataFrame(arr_t, columns = [c+'_tr' for c in cols])], axis = 1)
    #arr_t = power_transform(df.loc[:,cols])
    #return pd.concat([df, pd.DataFrame(arr_t, columns = [c+'_tr' for c in cols])], axis = 1)


def add_rolling_mean(df, filter=None, rev_filter=None, window=60):
    if filter is not None:
        df = df.filter(regex=filter)
    if rev_filter is not None:
        df = df.drop(df.filter(regex=rev_filter).columns, axis=1)
    arr_mean = df.rolling(window, center=True, min_periods=1).mean()#.add_suffix("_mean") 
    return arr_mean #pd.concat([df, arr_mean], axis = 1) #arr_mean

def select_features(df, names):
    if isinstance(names, str):
        if names == 'all':
            names = df.columns
    return df.loc[:, names]

def addhistory(df, dt_shifted, name_filter=None):
    """
    Concats history along axis 1 of df.
    Args: 
        df (DataFrame): dataframe across which to append history
        dt_shifted (list): list of absolute frame values to go back and forward in time, for each frame
        name_filter (list): column names that should not be included in addhistory, default None
    Returns:
        multshift (dataframe): dataframe with history, history columns have suffic _pos or _neg
    """
    org_df = df.copy()
    if name_filter is not None:
        name_filter_all = list(df.filter(regex="|".join(name_filter)).columns)
        name_keep = df.columns.drop(name_filter_all)
        #df_ = df.loc[:, name_filter_all]
        df = df.loc[:, name_keep]
    multshift = pd.DataFrame([])
    for i in dt_shifted:
        p_shift = df.shift(i)
        n_shift = df.shift(-i)
        multshift = pd.concat([multshift, p_shift.add_suffix(f"_pos{i}"), n_shift.add_suffix(f"_neg{i}")], axis=1)
    #if name_filter is not None:
    multshift = pd.concat([org_df, multshift], axis=1)
    
    return multshift

class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, step=30):
        self.step = step
        self.offset = offset
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X[30+self.step-1:-30:self.step]
        self.sample_index = X.index
        return X # check if this does not downsample during predictions
    def fit_transform(self, X, y=None):
        self.fit()
        X = self.transform(X,y)
        return X

class Smooth(BaseEstimator, TransformerMixin):
    def __init__(self, filter=None, rev_filter=None, window=60, ywindow=30, add=False, center=False):
        self.filter = filter
        self.rev_filter = rev_filter
        self.window = window
        self.ywindow = ywindow
        self.add = add
        self.center = center
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.filter is not None:
            X_ = X.filter(regex=self.filter)
            cols = X_.columns
        if self.rev_filter is not None:
            X_ = X.drop(X.filter(regex=self.rev_filter).columns, axis=1)
            cols = X_.columns
        else:            
            X_ = X.copy()
            cols = X_.columns
        X_m = X_.rolling(self.window, min_periods=self.window, center=self.center).mean()
        if self.add:
            X = pd.concat([X,X_m.add_suffix("_mean")], axis=1)
        else:
            X[cols] = X_m[cols]
        return X
    def fit_transform(self, X, y=None):
        X = self.transform(X)
        #if y is not None:
        #    y = y.rolling(self.ywindow, min_periods=self.ywindow, center=self.center).apply(lambda s: s.mode()[0])
        #    return X, y
        return X


