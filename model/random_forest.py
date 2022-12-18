from conf.conf import logging
from sklearn.ensemble import RandomForestRegressor


def create_rf_model(max_depth):
    logging.info(f"Creating random forest model with max_depth={max_depth}")
    rf_mean = RandomForestRegressor(max_depth=max_depth)
    return rf_mean
