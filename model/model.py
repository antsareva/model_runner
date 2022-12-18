from model.data_clearing_transformation import get_train_data
from util.util import *
from model.linear_regression import create_lin_reg_model
from model.random_forest import create_rf_model
from conf.conf import logging


def generate_model(dumped_model: str, model_name: str, dump_model_path: str, max_depth: int):
    model = None
    if dumped_model:
        model = load_model(dumped_model)
        logging.info(f"Loaded model from file {dumped_model}")
    else:
        if model_name == 'linear_regression':
            model = create_lin_reg_model()
        elif model_name == 'random_forest':
            model = create_rf_model(max_depth)
        if dump_model_path:
            save_model(dump_model_path, model)
            logging.info(f"Model saved to {dump_model_path}")
    return model


def run(dumped_model, model_name, dump_model_path, max_depth, test_size):
    model = generate_model(dumped_model, model_name, dump_model_path, max_depth)
    X_train, y_train, X_test, y_test = get_train_data(test_size)
    logging.info("Fitting model")
    model.fit(X_train, y_train)
    logging.info("Model fitted")
    logging.info("Predicting")
    y_pred = model.predict(X_test)
    logging.info("Predicted")
    logging.info(y_pred)
