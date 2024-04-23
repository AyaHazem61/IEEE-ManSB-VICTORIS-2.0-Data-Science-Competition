import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


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
from sklearn.base import BaseEstimator, TransformerMixin

class Target_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, target = 'cost', method = ['mean'], random_state = 42):
        self.random_state = random_state
        self.cols = cols
        self.target = target
        self.method = method
        self.cats_names = dict()
        self.cats_encoded = dict()
        self.encoding = dict()
        
    def fit(self, X, y = None):
        X_temp = X.copy()
        X_temp[self.target] = y
        for i, col in enumerate(self.cols):
            cats_mapped = X_temp.groupby(col)[self.target].mean()
            if self.method[i] == 'mean':
                cats_mapped = X_temp.groupby(col)[self.target].mean()
            elif self.method[i] == 'median':
                cats_mapped = X_temp.groupby(col)[self.target].median()
            self.cats_names[col] = cats_mapped.index
            self.cats_encoded[col] = cats_mapped.values
            self.encoding[col] = {self.cats_names[col][j] : self.cats_encoded[col][j] for j in range(len(cats_mapped))}        
        return self
    
    def transform(self, X):
        temp = X.copy()
        for col in self.cols:
            temp[col] = temp[col].map(self.encoding[col])
        return temp
    
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
    


class KNNImputerDF(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors = n_neighbors)
        
    def fit(self, X, y = None):
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        temp = self.imputer.transform(X)
        return pd.DataFrame(temp, columns = self.imputer.feature_names_in_)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

