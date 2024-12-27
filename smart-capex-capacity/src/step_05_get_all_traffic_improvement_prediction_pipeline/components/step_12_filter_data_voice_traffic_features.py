from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def filter_data_voice_traffic_features(project_id:str,
                                       location:str,
                                       data_traffic_features_table_id:str,
                                       voice_traffic_features_table_id:str,
                                       merged_data_traffic_features_data_input:Input[Dataset],
                                       merged_voice_traffic_features_data_input:Input[Dataset],
                                       data_traffic_features_data_output:Output[Dataset],
                                       voice_traffic_features_data_output:Output[Dataset]):
    """Filter data and voice traffic features and save the results as parquet files for current vertex pipeline
        and to BigQuery for the other pipelines.

    Args:
        project_id (str): GCP project ID.
        location (str): GCP location.
        data_traffic_features_table_id (str): Table ID for data traffic features.
        voice_traffic_features_table_id (str): Table ID for voice traffic features.
        merged_data_traffic_features_data_input (Input[Dataset]): Input dataset containing merged data traffic features.
        merged_voice_traffic_features_data_input (Input[Dataset]): Input dataset containing merged voice traffic features.
        data_traffic_features_data_output (Output[Dataset]): Output dataset to store filtered data traffic features.
        voice_traffic_features_data_output (Output[Dataset]): Output dataset to store filtered voice traffic features.
    """

    import pandas as pd
    import pandas_gbq

    df_merged_data_traffic_features = pd.read_parquet(merged_data_traffic_features_data_input.path)
    df_merged_voice_traffic_features = pd.read_parquet(merged_voice_traffic_features_data_input.path)


    mask_data_cluster_key = ~df_merged_data_traffic_features["cluster_key"].str.contains("202001")
    mask_voice_cluster_key = ~df_merged_voice_traffic_features["cluster_key"].str.contains("202001")

    df_data_traffic_features = df_merged_data_traffic_features[mask_data_cluster_key]
    df_voice_traffic_features = df_merged_voice_traffic_features[mask_voice_cluster_key]


    df_data_traffic_features.to_parquet(data_traffic_features_data_output.path)
    df_voice_traffic_features.to_parquet(voice_traffic_features_data_output.path)

    pandas_gbq.to_gbq(df_data_traffic_features,
                      data_traffic_features_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists='replace')

    pandas_gbq.to_gbq(df_voice_traffic_features,
                      voice_traffic_features_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists='replace')
