from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)

from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def prepare_dataset_of_future_upgrades(type_of_traffic: str,
                                       dict_traffic_improvement: dict,
                                       traffic_features_future_upgrades_data_input: Input[Dataset],
                                       x_data_output: Output[Dataset]):
    """Prepares the dataset of future upgrades for prediction.

    Args:
        type_of_traffic (str): It holds the type of traffic (data or voice).
        dict_traffic_improvement (dict): It holds traffic improvement information
        traffic_features_future_upgrades_data_input (Input[Dataset]): It holds the features of future upgrade
        x_data_output (Output[Dataset]): It holds the features for predictions.

    Returns:
        x_data_output (Output[Dataset]): It holds the features for predictions.
    """

    import pandas as pd

    df_traffic_features_future_upgrades = pd.read_parquet(traffic_features_future_upgrades_data_input.path)

    df_traffic_features_future_upgrades = df_traffic_features_future_upgrades[
        ~(df_traffic_features_future_upgrades["bands_upgraded"].isna())]
    df_traffic_features_encoded = pd.get_dummies(
        df_traffic_features_future_upgrades[["tech_upgraded", "bands_upgraded"]])

    df_traffic_features_future_upgrades = df_traffic_features_future_upgrades.join(
        df_traffic_features_encoded)
    df_traffic_features_future_upgrades.drop(columns=["tech_upgraded", "bands_upgraded"],
                                             inplace=True)

    cols_to_consider = [col for col in dict_traffic_improvement[type_of_traffic.upper() + "_TRAFFIC_FEATURES"]
                        if col in df_traffic_features_future_upgrades.columns]
    missing_cols = set(dict_traffic_improvement[type_of_traffic.upper() + "_TRAFFIC_FEATURES"]) - set(cols_to_consider)
    df_x = df_traffic_features_future_upgrades[cols_to_consider]
    df_x.fillna(-1, inplace=True)
    df_x[list(missing_cols)] = 0

    df_x.to_parquet(x_data_output.path)
