"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""


import pandas as pd
from sklearn.model_selection import train_test_split


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
    [train, test] = train_test_split(data, test_size=0.3, random_state=42)

    return train, test#, valid


