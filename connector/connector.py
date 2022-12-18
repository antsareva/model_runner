import pandas as pd
from conf.conf import logging
from conf.conf import settings


def get_data() -> pd.DataFrame:
    """
    This function extracts data from the source stated in the settings file
    """
    logging.info('Extracting df')
    data = pd.read_csv(settings.DATA_LINK)
    logging.info('Df is extracted')

    return data
