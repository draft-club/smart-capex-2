from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def post_process_upgraded_bands(project_id: str,
                                location: str,
                                selected_band_per_site_table_id: str,
                                week_of_the_upgrade: str,
                                congestion_data_input: Input[Dataset],
                                no_congestion_data_input: Input[Dataset],
                                congestion_status_data_output: Output[Dataset],
                                selected_band_per_site_data_output: Output[Dataset]):
    """It post processes the upgraded bands data

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        selected_band_per_site_table_id (str): It holds the resource name on BigQuery
        week_of_the_upgrade (str): It holds the week of the upgrade value
        congestion_data_input (Input[Dataset]): It holds the congestion data
        no_congestion_data_input (Input[Dataset]): It holds the no congestion data
        congestion_status_data_output (Output[Dataset]): It holds the concatenated congestion and no congestion cells
        selected_band_per_site_data_output (Output[Dataset]): It holds the post processed data

    Returns:
       congestion_status_data_output (Output[Dataset]): It holds the concatenated congestion and no congestion cells
       selected_band_per_site_data_output (Output[Dataset]): It holds the post processed data
    """

    # imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    df_congestion = pd.read_parquet(congestion_data_input.path)
    df_no_congestion = pd.read_parquet(no_congestion_data_input.path)

    df_congestion_status_concated = pd.concat([df_congestion, df_no_congestion]).reset_index(drop=True)

    df_selected_band_per_site = df_congestion_status_concated[df_congestion_status_concated["congestion"].isin(["3G_CONGESTION", "4G_CONGESTION"])]
    df_selected_band_per_site = df_selected_band_per_site[['site_id', 'congestion', 'cell_tech_available', 'tech_upgraded', 'bands_upgraded']]
    df_selected_band_per_site['week_of_the_upgrade'] = int(week_of_the_upgrade)

    df_selected_band_per_site = df_selected_band_per_site.drop_duplicates()

    df_selected_band_per_site.dropna(subset=["bands_upgraded"], inplace=True)
    print("df_selected_band_per_site after: ", df_selected_band_per_site.shape)

    df_selected_band_per_site.to_parquet(selected_band_per_site_data_output.path)
    df_congestion_status_concated.to_parquet(congestion_status_data_output.path)

    pandas_gbq.to_gbq(df_selected_band_per_site, selected_band_per_site_table_id,
                      project_id=project_id, location=location, if_exists='replace')
