from conf.conf import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from conf.conf import settings


def replace_with_mean(in_data, col_name):
    a = in_data.groupby([col_name])['Profit'].mean().sort_values(ascending=False)
    mean_dict = a.to_dict()
    in_data[f'{col_name}_mean'] = in_data[col_name].map(mean_dict)
    in_data = in_data.drop([col_name], axis=1)
    return in_data

def get_train_data(test_size: int):
    logging.info(f"Generating train&test data with test_size={test_size}")
    data = pd.read_csv(settings.DATA_LINK)

    # todo: filter columns from settings
    categorical_deatures = ['Ship Mode', 'Segment', 'City', 'State', 'Region', 'Category', 'Sub-Category']

    data_mean = data
    for i in categorical_deatures:
        data_mean = replace_with_mean(data_mean, i)
    data_mean = data_mean.drop(['State_mean'], axis=1)
    data_mean = data_mean.drop(['Country'], axis=1)
    logging.debug(data_mean.head())

    train_data_mean, test_data_mean = train_test_split(data_mean, test_size=test_size, random_state=settings.SEED)
    logging.debug(f"Train dataset shape (X): {train_data_mean.shape}")
    logging.debug(f"Test dataset shape (Y): {test_data_mean.shape}")


    X_train_mean = train_data_mean.loc[:, train_data_mean.columns != settings.PROFIT]
    y_train_mean = train_data_mean.loc[:, train_data_mean.columns == settings.PROFIT]
    X_test_mean = test_data_mean.loc[:, test_data_mean.columns != settings.PROFIT]
    y_test_mean = test_data_mean.loc[:, test_data_mean.columns == settings.PROFIT]

    return X_train_mean, y_train_mean, X_test_mean, y_test_mean

