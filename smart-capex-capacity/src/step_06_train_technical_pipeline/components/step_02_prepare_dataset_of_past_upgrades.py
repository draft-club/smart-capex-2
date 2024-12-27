from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def prepare_dataset_of_past_upgrades(type_of_traffic: str,
                                     remove_samples_with_target_variable_lower: bool,
                                     dict_technical_pipeline: dict,
                                     traffic_features_data_input: Input[Dataset],
                                     x_data_output: Output[Dataset],
                                     y_data_output: Output[Dataset]):
    """Prepare dataset of past upgrades and save the resulted dataframe to a parquet files for vertex pipeline.

    Args:
        type_of_traffic (str): It holds the type of traffic (e.g., "data" or "voice").
        remove_samples_with_target_variable_lower (bool): It holds the flag to remove samples with target variable
                                                            lower than a threshold.
        dict_technical_pipeline (dict): It holds the dictionary containing technical pipeline configurations.
        traffic_features_data_input (Input[Dataset]): It holds the input dataset containing traffic features.
        x_data_output (Output[Dataset]): It holds the output dataset to store the feature set in a dataframe.
        y_data_output (Output[Dataset]): It holds the output dataset to store the target variable dataframe.
    """

    import pandas as pd

    df_traffic_features = pd.read_parquet(traffic_features_data_input.path)

    df_traffic_features = df_traffic_features[~(df_traffic_features["bands_upgraded"].isna())]
    df_traffic_features_encoded = pd.get_dummies(df_traffic_features[["tech_upgraded", "bands_upgraded"]])
    df_traffic_features = df_traffic_features.join(df_traffic_features_encoded)
    df_traffic_features.drop(columns=["tech_upgraded", "bands_upgraded"], inplace=True)

    if remove_samples_with_target_variable_lower:
        feature = None
        if type_of_traffic == "data":
            feature = "total_data_traffic_dl_gb"
        elif type_of_traffic == "voice":
            feature = "total_voice_traffic_kerlands"

        df_traffic_features["is_higher"] = (df_traffic_features[feature + "_mean"]
                                            > df_traffic_features["target_variable"])
        df_traffic_features = df_traffic_features[~df_traffic_features["is_higher"]]
        df_traffic_features.drop(columns=["is_higher"], inplace=True)

    cols_to_consider = [col for col in
                        dict_technical_pipeline[type_of_traffic.upper() + "_TRAFFIC_FEATURES"]
                        if col in df_traffic_features.columns]

    missing_cols = set(dict_technical_pipeline[type_of_traffic.upper() + "_TRAFFIC_FEATURES"]) - set(cols_to_consider)

    df_x = df_traffic_features[cols_to_consider]
    df_x = df_x.fillna(-1)
    df_x[list(missing_cols)] = 0
    df_y = df_traffic_features[["target_variable"]]

    df_x.to_parquet(x_data_output.path, index=False)
    df_y.to_parquet(y_data_output.path, index=False)
