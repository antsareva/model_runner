from conf.conf import logging
from sklearn.linear_model import LinearRegression
from model.data_clearing_transformation import X_train_mean, y_train_mean, X_test_mean, y_test_mean



lin_reg_mean = LinearRegression()
lin_reg_mean.fit(X_train_mean, y_train_mean)
name = 'train_mean_lr'
y_pred_train_mean = lin_reg_mean.predict(X_train_mean)
# plot_predictions(y_train_mean, y_pred_train_mean, title = f'Comparison of y and y_pred for X_{name}')
name = 'test_mean_lr'
y_pred_test_mean = lin_reg_mean.predict(X_test_mean)
logging.info(f"Y prediction on test dataset: {y_pred_test_mean}")
# plot_predictions(y_test_mean, y_pred_test_mean, title = f'Comparison of y and y_pred for X_{name}')