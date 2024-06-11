#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle

import pandas as pd
import sklearn
# import flask
from flask import Flask, Response, jsonify, request
from loguru import logger

app = Flask(__name__)


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


@app.route('/ping', methods=['GET'])
async def ping() -> dict:
    return Response(
        jsonify(dict(response="pong", status=200, mimetype='application/json'))
    )


def read_data(filename) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    logger.debug(f"\n\t [i] Reading file: {filename}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def predict(year=2023, month=4, taxi="yellow") -> pd.DataFrame:

    df = read_data(
        f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year:04d}-{month:02d}.parquet')
    logger.debug(
        f"\n\t [i] Successfully loaded {taxi} taxi data {year:04d}-{month:02d}..")

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    logger.debug(f"]\n\t [i] Successfully ran predictions (column: predict).")

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + \
        df.index.astype('str')
    df_result['predict'] = y_pred

    return df_result


# async def predictor() -> None:
#     # logger.debug(f"ARGS: {args}")
#     logger.debug(f"REQUEST: {request.get_json()}")
#     return Response(
#         dict(response=request.get_json(), status=200)
#     )

@logger.catch  # type: ignore
@app.route('/predict', methods=['POST'])
async def infer() -> None:
    args = request.get_json()
    year = args.get('year', 2023)
    month = args.get('month', 4)
    taxi = args.get('taxi', 'yellow')

    df_result = predict(year=year, month=month, taxi=taxi)
    logger.debug(
        f"\n\t [i] Received result pd.DataFrame with shape : {df_result.shape}")

    output_folder = os.path.join('output', taxi)
    output_file = os.path.join(
        output_folder, f'result_{year:04d}-{month:02d}.parquet')

    if bucket := args.get('bucket', None):
        fs = s3fs.S3FileSystem()
        df.to_parquet(
            f's3://{bucket}/{output_file}',
            engine='pyarrow',
            filesystem=fs,
            compression=None,
            index=False
        )

    logger.debug(
        f"\n\t [i] SUCCESS! The average of prediction results is : {df_result.loc[:, 'predict'].mean()}")

    return Response(dict(
        status=200,
        result=df_result.loc[:, 'predict'].mean()
    ))  # type: ignore
