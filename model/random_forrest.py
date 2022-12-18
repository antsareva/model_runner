from conf.conf import logging
from sklearn.ensemble import RandomForestRegressor
from model.data_clearing_transformation import X_train_mean, y_train_mean, X_test_mean, y_test_mean


# Гиперпараметр подобран, максимальная глубина равна 14, на ней достигается минимальный rmse
rf_mean = RandomForestRegressor(max_depth=14)
rf_mean.fit(X_train_mean, y_train_mean)

y_pred = rf_mean.predict(X_train_mean)

y_pred = rf_mean.predict(X_test_mean)
logging.info(f"Y prediction on test dataset: {y_pred}")
