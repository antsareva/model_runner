from conf.conf import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from conf.conf import settings
from connector.connector import get_data


def replace_with_mean(in_data, col_name):
    a = in_data.groupby([col_name])[settings.TARGET].mean().sort_values(ascending=False)
    mean_dict = a.to_dict()
    in_data[f'{col_name}_mean'] = in_data[col_name].map(mean_dict)
    in_data = in_data.drop([col_name], axis=1)
    return in_data

def intersection(lst1: list, lst2: list) -> list:
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_train_data(test_size: int):
    data = get_data()[settings.FEATURES]
    categorical_features = ['Ship Mode', 'Segment', 'City', 'State', 'Region', 'Category', 'Sub-Category']
    categorical_features = intersection(settings.FEATURES, categorical_features)
    logging.info(f"Generating train&test data with test_size={test_size}")
    data_mean = data
    for i in categorical_features:
        data_mean = replace_with_mean(data_mean, i)
    logging.debug(data_mean.head())

    train_data_mean, test_data_mean = train_test_split(data_mean, test_size=test_size, random_state=settings.SEED)
    logging.debug(f"Train dataset shape: {train_data_mean.shape}")
    logging.debug(f"Test dataset shape: {test_data_mean.shape}")


    X_train_mean = train_data_mean.loc[:, train_data_mean.columns != settings.TARGET]
    y_train_mean = train_data_mean.loc[:, train_data_mean.columns == settings.TARGET]
    X_test_mean = test_data_mean.loc[:, test_data_mean.columns != settings.TARGET]
    y_test_mean = test_data_mean.loc[:, test_data_mean.columns == settings.TARGET]

    return X_train_mean, y_train_mean, X_test_mean, y_test_mean

