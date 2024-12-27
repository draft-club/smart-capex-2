from kfp.dsl import (Dataset, Input, component)
from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def merge_congestion_with_sites(project_id: str,
                                location: str,
                                congestion_status_for_db_table_id: str,
                                congestion_status_data_input: Input[Dataset],
                                sites_data_input: Input[Dataset]):
    """It merges the concatenated congestiion and no congestion data with the sites data

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        congestion_status_for_db_table_id (str): It holds the resource name on BigQuery
        congestion_status_data_input (Input[Dataset]): It holds the concatenated congestiion and no congestion data
        sites_data_input (Input[Dataset]): It holds the sites data

    Returns:
        None
    """

    # imports
    import pandas as pd
    import pandas_gbq

    def add_nb_congested_cells(df):
        df["nb_3g_congested_cells"] = df[df["congestion"] == "3G_CONGESTION"]["cell_name"].nunique()
        df["nb_4g_congested_cells"] = df[df["congestion"] == "4G_CONGESTION"]["cell_name"].nunique()
        df["total_nb_congested_cells"] = df["nb_3g_congested_cells"] + df["nb_4g_congested_cells"]
        return df

    # Load Data
    df_congestion_status = pd.read_parquet(congestion_status_data_input.path)
    df_sites = pd.read_parquet(sites_data_input.path)

    df_congestion_status = df_congestion_status.groupby(['week_date', 'year', 'week',
                                                         'week_period', 'site_id', 'congestion']
                                                        ).apply(add_nb_congested_cells)
    df_sites_info = df_sites[["cell_name", "region", "latitude", "longitude"]]

    df_sites_info.rename(columns={"latitude": "site_latitude", "longitude": "site_longitude"}, inplace=True)
    df_congestion_for_db = pd.merge(left=df_congestion_status, right=df_sites_info, on="cell_name")

    # Join for saving in database as string object
    df_congestion_for_db['cell_band_available'] = df_congestion_for_db['cell_band_available'].apply(lambda x: '_'.join(tuple(x)))

    pandas_gbq.to_gbq(df_congestion_for_db, congestion_status_for_db_table_id, project_id=project_id,
                      location=location, if_exists='replace')
