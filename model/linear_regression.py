from conf.conf import logging
from sklearn.linear_model import LinearRegression

def create_lin_reg_model():
    logging.info("Creating linear regression model")
    lin_reg_mean = LinearRegression()
    return lin_reg_mean
