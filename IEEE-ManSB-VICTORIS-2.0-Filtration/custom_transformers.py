import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""

    A transformer that performs winsorization imputation on specified columns in a Pandas DataFrame.

    Parameters:
    -----------
    p : float, default=0.05
        The percentile value representing the lower bound for winsorization.
    q : float, default=0.95
        The percentile value representing the upper bound for winsorization.
    random_state : int, default=42
        Seed for the random number generator used for imputing missing values.
    cols : list, default=None
        The list of names of columns to be winsorized.

    Returns:
    --------
    A new Pandas DataFrame with the specified columns winsorized.

 """

class WinsorizationImpute(BaseEstimator, TransformerMixin):

    def __init__(self, p=0.05, q=0.95, random_state=42, cols=None):
        self.p = p
        self.q = q
        self.random_state = random_state
        self.cols = cols
        
    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols:
            lower_bound = X[col].quantile(self.p)
            upper_bound = X[col].quantile(self.q)
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
        return self
    
    def transform(self, X):
        X_winsorized = X.copy()
        for col in self.cols:
            lower_bound = self.lower_bounds_[col]
            upper_bound = self.upper_bounds_[col]
            outliers_mask = (X_winsorized[col] < lower_bound) | (X_winsorized[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                random_values = np.random.normal(loc=X_winsorized[col].mean(),
                                                 scale=X_winsorized[col].std(),
                                                 size=outliers_count)
                random_values = np.clip(random_values, lower_bound, upper_bound)
                X_winsorized.loc[outliers_mask, col] = random_values
        return X_winsorized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

    """
    A transformer that performs Target Encoding on specified column in a Pandas DataFrame.

    Parameters:
    -----------
    col : string
        The column name of the categorical variable you want to target encode
    target : string
        The percentile value representing the upper bound for winsorization.
    method : string, default='mean'
        If 'mean' categories are encoded by getting mean target
        If 'median' categories are encoded by getting median target
    random_state : int, default=42
        Seed for the random number generator used for imputing missing values.
    Returns:
    --------
    A new Pandas DataFrame with the specified columns winsorized.

"""
class Target_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, col, target = 'Cost', method = 'mean', random_state = 42):
        self.random_state = random_state
        self.col = col
        self.target = target
        self.method = method
        
    def fit(self, X, y = None):
        X_temp = X.copy()
        X_temp[self.target] = y
        if self.method == 'mean':
            cats_mapped = X_temp.groupby(self.col)[self.target].mean()
        elif self.method == 'median':
            cats_mapped = X_temp.groupby(self.col)[self.target].median()
            
        self.cats_names = cats_mapped.index
        self.cats_encoded = cats_mapped.values
        self.encoding = {self.cats_names[i] : self.cats_encoded[i] for i in range(len(cats_mapped))}
        
        return self
    
    def transform(self, X):
        return X[self.col].map(self.encoding).values.reshape(-1, 1)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

"""
    A transformer that performs Multi-Hot Encoding on specified column in a Pandas DataFrame.

    Input should be Pandas DataFrame with columns of lists of categories

"""
class Multi_Hot_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y = None):
        classes__ = set()
        X_temp = X.copy()
        for row in X[self.col]:
            for cat in row:
                classes__.add(cat)
        classes__ = list(classes__)
        self.nclasses__ = len(classes__)
        self.classes__ = classes__
        return self
    
    def transform(self, X):
        results = np.zeros((X.shape[0], self.nclasses__))
        for i in range(self.nclasses__):
            results[:, i] = X[self.col].apply(lambda x: self.classes__[i] in x)
        return results
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
