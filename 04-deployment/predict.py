#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle

import pandas as pd
import sklearn
from loguru import logger

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    if args.debug:
        logger.debug(f"\n\t [i] Reading file: {filename}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def predict(args) -> pd.DataFrame:

    f_input = os.path.join(
        'data', f'{args.taxi}_tripdata_{args.year:04d}-{args.month:02d}.parquet')
    if os.path.exists(f_input):
        df = read_data(f_input)
    else:
        df = read_data(
            f'https://d37ci6vzurychx.cloudfront.net/trip-data/{args.taxi}_tripdata_{args.year:04d}-{args.month:02d}.parquet')
    if args.debug:
        logger.debug(
            f"\n\t [i] Successfully loaded {args.taxi} taxi data {args.year:04d}-{args.month:02d}..")

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    if args.debug:
        logger.debug(f"]\n\t [i] Successfully ran predictions..")

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + \
        df.index.astype('str')
    df_result['predict'] = y_pred

    return df_result


def main(args) -> None:
    output_folder = os.path.join('output', args.taxi)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(
        output_folder, f'result_{args.year:04d}-{args.month:02d}.parquet')
    df_result = predict(args)
    logger.debug(
        f"\n\t [i] Received result pd.DataFrame with shape : {df_result.shape}")

    if args.bucket:
        fs = s3fs.S3FileSystem()
        df.to_parquet(
            f's3://{args.bucket}/{output_file}',
            engine='pyarrow',
            filesystem=fs,
            compression=None,
            index=False
        )

    else:  # write to local folder
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

    if args.average:
        logger.debug(
            f"\n\t [i] DEBUG: Standard deviation of predictions is {df_result.loc[:, 'predict'].mean()}")
    if args.debug:
        logger.debug(f"\n\t [i] SUCCESS! Wrote {output_file}")


def cli_params() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug",
                        required=False, action='store_true',
                        help="Increase run-time verbosity.")

    parser.add_argument("-y", "--year",
                        required=True, type=int,
                        help="Year of NYC Taxi dataset.")

    parser.add_argument("-m", "--month",
                        required=True, type=int,
                        help="Month of NYC Taxi dataset.")

    parser.add_argument("-t", "--taxi",
                        required=False, default="yellow",
                        help="Specific color of NYC Taxi.", )

    parser.add_argument("-b", "--bucket",
                        required=False, type=str,
                        help="(Optional) Write output to S3 Bucket.")

    parser.add_argument("-a", "--average", "--avg",
                        required=False, action='store_true',
                        help="Summarize output predicted average (mean) duration.")

    args = parser.parse_args()

    if args.debug:
        logger.debug(f"ARGS: {args}")

    return (args)


if __name__ == "__main__":
    args = cli_params()
    main(args)
