from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_distance_between_sites(processed_sites_data_input: Input[Dataset],
                                   sites_distances_data_output: Output[Dataset]):
    """It computes the distance between sites

    Args:
        processed_sites_data_input (Input[Dataset]): It holds the processed sites data
        sites_distances_data_output (Output[Dataset]): It holds the distance between sites

    Returns:
        sites_distances_data_output (Output[Dataset]): It holds the distance between sites
    """

    # Imports
    import pandas as pd
    import haversine as hs
    from scipy import spatial

    # Load Data
    df_sites = pd.read_parquet(processed_sites_data_input.path)

    df_sites = df_sites[["longitude", "latitude", "site_id"]].drop_duplicates()
    df_sites = df_sites.dropna(subset=['latitude']).drop_duplicates()
    df_sites = df_sites.groupby("site_id").first().reset_index(drop=False)

    # Extract latitude and longitude values as an array
    site_coordinates = df_sites[["latitude", "longitude"]].values

    # Calculate pairwise distances between all points using the Haversine formula
    distance_matrix = spatial.distance.cdist(site_coordinates, site_coordinates, hs.haversine)

    # Create a DataFrame with the distance matrix, using site IDs as both index and column names
    df_distance_matrix = pd.DataFrame(distance_matrix, index=df_sites["site_id"].values, columns=df_sites["site_id"].values)

    print("df_distance_matrix shape after: ", df_distance_matrix.shape)

    df_distance_matrix.to_parquet(sites_distances_data_output.path)
