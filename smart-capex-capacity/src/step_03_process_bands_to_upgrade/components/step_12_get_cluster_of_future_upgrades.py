from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def get_cluster_of_future_upgrades(project_id: str,
                                   location: str,
                                   cluster_of_the_upgrade_table_id: str,
                                   week_of_the_upgrade: str,
                                   max_number_of_neighbors: int,
                                   sites_distances_data_input: Input[Dataset],
                                   selected_band_per_site_data_input: Input[Dataset],
                                   cluster_of_the_upgrade_data_output: Output[Dataset]):
    """It

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        cluster_of_the_upgrade_table_id (str): It holds the resource name on BigQuery
        week_of_the_upgrade (str): It holds the week of the upgrade value
        max_number_of_neighbors (int): It holds the maximum number of neighbors value
        sites_distances_data_input (Input[Dataset]): It holds the sites data
        selected_band_per_site_data_input (Input[Dataset]): It holds the selected band per site data
        cluster_of_the_upgrade_data_output (Output[Dataset]): It holds the cluster of upgrade data

    Returns:
        cluster_of_the_upgrade_data_output (Output[Dataset]): It holds the cluster of upgrade data
    """

    # imports
    import pandas as pd
    import pandas_gbq

    def compute_neighbour_site(site_id: pd.DataFrame,
                               df_distance: pd.DataFrame,
                               position: int) -> str:
        """It computes the neighbours of sites

        Args:
            site_id (pd.DataFrame): It holds the site_id
            df_distance (pd.DataFrame): It holds the distance between sites
            position (int): It holds the position of the site

        Returns:
            str: It holds the site_id
        """
        try:
            distance_vector = df_distance[site_id]
        except:
            return ""

        distance = 3.5
        distance_vector = distance_vector[(distance_vector > 0)
                                        & (distance_vector < distance)].sort_values()

        if position > len(distance_vector):
            return ""
        return distance_vector.index[position - 1]

    # Load Data
    df_distance = pd.read_parquet(sites_distances_data_input.path)
    df_selected_band_per_site = pd.read_parquet(selected_band_per_site_data_input.path)

    df_selected_band_per_site["cluster_key"] = df_selected_band_per_site["site_id"]

    # ToDo: should this for loop be removed as the max_numbers_of_neighbours is set to 0 in config
    df_selected_band_per_site["cluster_key"] = df_selected_band_per_site["site_id"]

    for i in range(max_number_of_neighbors):
        df_selected_band_per_site["neighbor_" + str(i + 1)] = df_selected_band_per_site[["site_id", "bands_upgraded"]].apply(lambda x: compute_neighbour_site(x.iloc[0], df_distance, i + 1), axis=1)
        df_selected_band_per_site["cluster_key"] = (df_selected_band_per_site["cluster_key"] + "_" + df_selected_band_per_site["neighbor_" + str(i + 1)].astype(str))

    df_selected_band_per_site['week_of_upgrade'] = int(week_of_the_upgrade)

    print("df_selected_band_per_site shape: ", df_selected_band_per_site.shape)
    print("df_selected_band_per_site info: ", df_selected_band_per_site.info())

    df_selected_band_per_site.to_parquet(cluster_of_the_upgrade_data_output.path)

    # Save to bigquery
    pandas_gbq.to_gbq(df_selected_band_per_site, cluster_of_the_upgrade_table_id,
                      project_id=project_id, location=location, if_exists='replace')
