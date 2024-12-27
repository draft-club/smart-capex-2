from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def compute_distance_between_sites(processed_sites_data_input:Input[Dataset],
                                   sites_distances_data_output:Output[Dataset]):
    """Calculates the distances between sites using their latitude and longitude.

    Args:
        processed_sites_data_input (Input[Dataset]): It holds the processed sites returned from BigQuery
        sites_distances_data_output (Output[Dataset]): It holds the distances between sites
        
    Returns:
        sites_distances_data_output (Output[Dataset]): It holds the distances between sites 
    """

    import pandas as pd
    import haversine as hs
    from scipy import spatial

    df_sites =  pd.read_parquet(processed_sites_data_input.path)
    df_sites = df_sites[["longitude", "latitude", "site_id"]].drop_duplicates()

    df_sites.dropna(subset=['latitude'], inplace=True)
    df_sites = df_sites.groupby("site_id").first().reset_index(drop=False)
    all_points = df_sites[["latitude", "longitude"]].values
    dm1 = spatial.distance.cdist(all_points, all_points, hs.haversine)

    df_distance = pd.DataFrame(dm1, index=df_sites["site_id"].values,
                               columns=df_sites["site_id"].values)

    df_distance.to_parquet(sites_distances_data_output.path)
