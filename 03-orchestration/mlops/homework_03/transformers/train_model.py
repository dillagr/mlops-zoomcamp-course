import pandas as pd

from sklearn.linear_model import LinearRegression
# from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer


# mlflow.sklearn.autolog()

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    dv = DictVectorizer()
    categorical = ['PULocationID', 'DOLocationID']
    # numerical = ['trip_distance']

    dicts = df[categorical].to_dict(orient='records')

    X = dv.fit_transform(dicts)
    y = df['duration'].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"Intercept: {model.intercept_}")

    return dv, model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'