import mlflow
import joblib

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "linear-regression-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "linear-regression-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    # client = MlflowClient()

    # experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # run = client.search_runs(
    #     experiment.experiment_id, order_by=["start_time DESC", "run_id"], max_results=1)[0]
    # mlflow.register_model(
    #     model_uri=f"run:/{best_run.info.run_id}/model", name="LinearRegressionModel")
    # print(data)
    dv, model = data
    ARTIFACT_FILE = "DictVectorizer.out"
    with open(ARTIFACT_FILE, 'wb') as f:
        joblib.dump(dv, f)

    with mlflow.start_run() as r:

        run_id = r.info.run_id
        mlflow.log_artifact(ARTIFACT_FILE)
        mlflow.sklearn.log_model(model, "LinearRegression")
