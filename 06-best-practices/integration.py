import os
from datetime import datetime

import pandas as pd
from icecream import ic

import batch

# icecream
ic.disable()


def save_data(df, filename):
    """
    (I/O part) Save the dataset to file.
    filename - filename of parquet file
    categorical - list of categorical features
    """

    if ic(endpoint := os.getenv('S3_ENDPOINT_URL')):
        options = {
            'client_kwargs': {
                'endpoint_url': endpoint,
            }
        }

        df.to_parquet(
            f's3://nyc-dataset-filestore/{filename}.parquet', storage_options=options)
    else:
        print(
            f" [!] ERROR: S3_ENDPOINT_URL not defined. Cannot save {filename}.")


def dt(hour, minute, second=0):
    """
    Helper function to convert to datetime
    hour - hour in terms of units of time, 24H format
    minute - minute in terms of units of time
    second - second in terms of units of time
    """
    return datetime(2023, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']

df = ic(pd.DataFrame(data, columns=columns))


df_output = ic(batch.prepare_data(
    df, categorical=['PULocationID', 'DOLocationID']))


save_data(df_output, 'integration.parquet')
