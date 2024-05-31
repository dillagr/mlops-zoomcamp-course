import pandas as pd
from sklearn.feature_extraction import DictVectorizer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def fit_transform(df: pd.DataFrame, *args, **kwargs):
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

    # categorical = ['PULocationID', 'DOLocationID']
    # numerical = ['trip_distance']
    # dicts = df[categorical + numerical].to_dict(orient='records')

    dicts = df['PULocationID', 'DOLocationID', 'trip_distance'].to_dict(orient='records')

    X = dv.fit_transform(dicts)
    y = df['duration'].values

    return dv, X, y


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'