"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.signal import savgol_filter


def processing(data: pd.DataFrame):
    """
    It takes a dataframe and returns a tuple of two dataframes

    Args:
      data (pd.DataFrame): the dataframe to split

    Returns:
      Three dataframes.
    """
    # data = data.drop(columns=['IDH'])
    data_clean = data.dropna(subset=['group'])
    data_filled = data_clean.fillna(0)

    return data_clean, data_filled
    

def split_train_test_valid(data: pd.DataFrame):
    """
    It takes a dataframe and returns a tuple of two dataframes

    Args:
      data (pd.DataFrame): the dataframe to split

    Returns:
      Three dataframes.
    """
    [train, test_valid] = train_test_split(data, test_size=0.2, random_state=42)
    [test, valid] = train_test_split(test_valid, test_size=0.5, random_state=42)

    return train, test, valid


def def_features_n_outcome(data: pd.DataFrame):
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = data.drop(columns=['Age', 'group'])
    Age = data['Age']
    y = data['group']

    return X, Age, y

def smoother(X, window=11, order=2):
    
  X_smooth=savgol_filter(np.array(X),window,order)
  
  return X_smooth


def scaler(X, scaler= None):
     
  if scaler is None:
    scaler=preprocessing.MinMaxScaler()

  X_scaled=scaler.fit_transform(X)

  return pd.DataFrame(X_scaled),scaler


def transformer(X, transformer=None):

  if transformer is None:
    transformer=preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)

  X_tr=transformer.fit_transform(X)

  return X_tr, transformer


def normalizer(X,norm='l2'):  
    
  X_norm=preprocessing.normalize(X,norm=norm)
  
  return X_norm



