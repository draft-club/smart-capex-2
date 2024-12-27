from kfp.dsl import Dataset, Input, Model, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def train_traffic_improvement_model(type_of_traffic: str,
                                    model_path: str,
                                    exec_time: str,
                                    x_data_input: Input[Dataset],
                                    y_data_input: Input[Dataset],
                                    model_output: Output[Model]):
    """Train traffic improvement XGBoost regression model and save the model artifact to Cloud Storage.

    Args:
        type_of_traffic (str): It holds the type of traffic (e.g., "data" or "voice").
        model_path (str): It holds the path to save the model.
        exec_time (str): It holds the execution time for versioning the model.
        x_data_input (Input[Dataset]): It holds the input dataset containing the feature set (X).
        y_data_input (Input[Dataset]): It holds the input dataset containing the target variable (Y).
        model_output (Output[Model]): It holds the output model artifact.
    """

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import storage
    from xgboost import XGBRegressor

    df_x = pd.read_parquet(x_data_input.path)
    df_y = pd.read_parquet(y_data_input.path)

    x_train = np.array(df_x)
    y_train = np.array(df_y["target_variable"])

    model_parameters = {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 50, 'subsample': 0.5}
    model = XGBRegressor(**model_parameters)
    model.fit(x_train, y_train)

    model_filename = f"{type_of_traffic}_traffic_improvement_xgboost.sav"
    model_path = f"{model_path}/traffic_improvement_model/{exec_time}/{model_filename}"

    # Save model artifact to local filesystem (doesn't persist)
    local_filename = model_output.path + ".sav"
    with open(local_filename, 'wb') as file:
        joblib.dump(model, file)

    blob = storage.blob.Blob.from_string(model_path, client=storage.Client())
    blob.upload_from_filename(local_filename)
