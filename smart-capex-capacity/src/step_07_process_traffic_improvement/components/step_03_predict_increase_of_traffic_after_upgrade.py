from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)

from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def predict_increase_of_traffic_after_upgrade(gcs_bucket: str,
                                              xgboost_model_path: str,
                                              type_of_traffic: str,
                                              x_data_input: Input[Dataset],
                                              traffic_features_future_upgrades_data_input: Input[Dataset],
                                              traffic_features_future_upgrades_data_output: Output[Dataset]):
    """Predicts the increase of traffic after an upgrade.

    Args:
        gcs_bucket (str): It holds the name of the GCS bucket.
        xgboost_model_path (str): It holds the path to the XGBoost model.
        type_of_traffic (str): It holds the type of traffic (data or voice).
        x_data_input (Input[Dataset]): It holds the features for predictions. 
        traffic_features_future_upgrades_data_input (Input[Dataset]): It holds future upgrade data.
        traffic_features_future_upgrades_data_output (Output[Dataset]): It holds the predictions and the increase of traffic
    Returns:
        traffic_features_future_upgrades_data_output (Output[Dataset]): It holds the predictions and the increase of traffic
    """

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import storage

    def load_model_from_gcs(bucket, model_path):
        # Load the model pickle file from cloud storage
        client = storage.Client()
        bucket = client.get_bucket(bucket)
        blob = bucket.blob(model_path)

        with blob.open("rb") as file:
            model = joblib.load(file)

        return model

    xgboost_model = load_model_from_gcs(gcs_bucket, xgboost_model_path)

    df_x = pd.read_parquet(x_data_input.path)
    df_traffic_features_future_upgrades = pd.read_parquet(traffic_features_future_upgrades_data_input.path)

    ### HINT: added this line to convert the input dataframe into an array
    ### as `prepare_dataset_of_future_upgrades` returns a DataFrame
    x = np.array(df_x)

    df_traffic_features_future_upgrades["predictions"] = xgboost_model.predict(x)

    if type_of_traffic == "data":
        df_traffic_features_future_upgrades["increase_of_traffic_after_the_upgrade"] = (
                df_traffic_features_future_upgrades["predictions"]
                - df_traffic_features_future_upgrades["total_data_traffic_dl_gb_mean"])
    elif type_of_traffic == "voice":
        df_traffic_features_future_upgrades["increase_of_traffic_after_the_upgrade"] = (
                df_traffic_features_future_upgrades["predictions"]
                - df_traffic_features_future_upgrades["total_voice_traffic_kerlands_mean"])

    df_traffic_features_future_upgrades["increase_of_traffic_after_the_upgrade"] = \
        df_traffic_features_future_upgrades["increase_of_traffic_after_the_upgrade"].clip(lower=0)

    df_traffic_features_future_upgrades.to_parquet(traffic_features_future_upgrades_data_output.path)
