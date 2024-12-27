from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def merge_predicted_kpis_with_bands(project_id: str,
                                    location: str,
                                    affected_cells_table_id: str,
                                    predicted_traffic_kpis_data_input: Input[Dataset],
                                    selected_band_per_site_data_input: Input[Dataset],
                                    merged_predicted_kpis_with_bands_data_output: Output[Dataset]):
    """It merges the predicted kpis with the upgraded bands data

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        affected_cells_table_id (str): It holds the resource name on BigQuery
        predicted_traffic_kpis_data_input (Input[Dataset]): It holds the predicted traffic kpis data
        selected_band_per_site_data_input (Input[Dataset]): It holds the selected band per site data
        merged_predicted_kpis_with_bands_data_output (Output[Dataset]): It holds the merged predicted kpis with bands data

    Returns:
        merged_predicted_kpis_with_bands_data_output (Output[Dataset]): It holds the merged predicted kpis with bands data
    """

    # imports
    import pandas as pd
    import pandas_gbq

    df_predicted_traffic_kpis = pd.read_parquet(predicted_traffic_kpis_data_input.path)
    df_selected_band_per_site = pd.read_parquet(selected_band_per_site_data_input.path)

    df_predicted_traffic_kpis = df_predicted_traffic_kpis[(df_predicted_traffic_kpis["average_number_of_users_in_queue"] != 9999) |
                                                          (df_predicted_traffic_kpis["average_throughput_user_dl"] != 9999)]

    df_affected_cells = df_predicted_traffic_kpis.merge(df_selected_band_per_site, on='site_id', how='inner')

    print("df_affected_cells shape: ", df_affected_cells.shape)
    print("df_affected_cells info: ", df_affected_cells.info())

    df_affected_cells.to_parquet(merged_predicted_kpis_with_bands_data_output.path)

    pandas_gbq.to_gbq(df_affected_cells, affected_cells_table_id, project_id=project_id,
                      location=location, if_exists='replace')
