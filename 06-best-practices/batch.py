#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle

import pandas as pd
from icecream import ic

# import sys


def get_input_path(year, month):
    """
    Input path for data
    """
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    """
    Output path for data
    """
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df, categorical):
    """
    (Transformation part) Prepare the dataset.
    df - a pandas.Dataframe
    categorical - a list of categorical features
    """

    # df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    # df['duration'] = df.duration.dt.total_seconds() / 60

    # df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def read_data(filename, categorical):
    """
    (I/O part) Read the dataset from file.
    filename - filename of parquet file
    categorical - list of categorical features
    """
    if ic(endpoint := os.getenv('S3_ENDPOINT_URL')):
        options = {
            'client_kwargs': {
                'endpoint_url': endpoint
            }
        }
        df = pd.read_parquet(
            f's3://nyc-dataset-filestore/{filename}.parquet', storage_options=options)
    else:
        df = pd.read_parquet(filename)

    df = prepare_data(df, categorical)

    return df


def main(year, month):
    """
    Main function
    year - year (numeric integer), in terms of units of time, 4-digit year
    month - month (numeric integer), in terms of units of time, 1-12 (1:January.. and so on)
    """

    # cf_url = 'https://d37ci6vzurychx.cloudfront.net'
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    x_val = dv.transform(dicts)
    y_pred = lr.predict(x_val)

    print('predicted mean duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


def cli_params():
    """
    Parse the arguments from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "-v",
        "--debug",
        required=False,
        action='store_true',
        help="Make verbose outputs on stdout (similar function as verbose).",
    )

    parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int,
        help="Year of the dataset.",
    )

    parser.add_argument(
        "-m",
        "--month",
        required=True,
        type=int,
        help="Month of the dataset.",
    )

    args = parser.parse_args()
    # if args.debug:
    #     logger.debug(f"ARGS: {args}")

    return args


if __name__ == "__main__":
    arguments = cli_params()
    main(year=arguments.year, month=arguments.month)
