"""
test_batch.py - unit tests for batch.py
"""

from datetime import datetime

import pandas as pd
# from loguru import logger
from pandas.api.types import is_string_dtype

import batch


def dt(hour, minute, second=0):
    """
    Helper function to convert to datetime
    hour - hour in terms of units of time, 24H format
    minute - minute in terms of units of time
    second - second in terms of units of time
    """
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    """
    test module for prepare_data
    """
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    # logger.debug(df.to_dict('records'))

    df_output = batch.prepare_data(
        df, categorical=['PULocationID', 'DOLocationID'])
    # logger.debug(df_output.to_dict('records'))

    assert len(df_output) <= len(df), "Data input not the same as output"
    assert is_string_dtype(
        df_output['PULocationID']), "PULocationID is not a string"
    assert is_string_dtype(
        df_output['DOLocationID']), "DOLocationID is not a string"
    assert df.isnull().values.any(), "DataFrame contains null values"
    assert "duration" in df_output.columns, "DataFrame has to have the duration column"
