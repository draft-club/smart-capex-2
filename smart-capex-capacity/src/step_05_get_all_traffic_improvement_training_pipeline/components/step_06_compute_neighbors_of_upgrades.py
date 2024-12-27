from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_neighbors_of_upgrades(max_number_of_neighbors: int,
                                  remove_sites_with_more_than_one_upgrade_same_cluster: bool,
                                  list_of_upgrades_data_input: Input[Dataset],
                                  sites_to_remove_data_input: Input[Dataset],
                                  neighbors_of_upgrades_data_output: Output[Dataset]):
    """Compute neighbors of upgrades and save the results as a parquet file for vertex pipeline.

    Args:
        max_number_of_neighbors (int): It holds the maximum number of neighbors.
        remove_sites_with_more_than_one_upgrade_same_cluster (bool): It holds the flag to remove sites with 
                                                                        more than one upgrade in the same cluster.
        list_of_upgrades_data_input (Input[Dataset]): It holds the input dataset containing the list of upgrades.
        sites_to_remove_data_input (Input[Dataset]): It holds the input dataset containing the sites to remove.
        neighbors_of_upgrades_data_output (Output[Dataset]): It holds the output dataset to store the neighbors of upgrades.
    """

    import pandas as pd

    df_list_of_upgrades = pd.read_parquet(list_of_upgrades_data_input.path)
    df_sites_to_remove = pd.read_parquet(sites_to_remove_data_input.path)

    sites_to_remove = df_sites_to_remove['site_id'].tolist()

    if remove_sites_with_more_than_one_upgrade_same_cluster:
        df_list_of_upgrades = df_list_of_upgrades[~df_list_of_upgrades["site_id"].isin(sites_to_remove)]

    list_neighbors = ["neighbor" + "_" + str(i + 1) for i in range(max_number_of_neighbors)]

    df_neighbors = pd.melt(df_list_of_upgrades[["cluster_key"] + list_neighbors],
                           id_vars=["cluster_key"],
                           value_vars=list_neighbors)

    df_neighbors.columns = ["cluster_key", "variable", "neighbor_name"]
    df_neighbors = df_neighbors[~(df_neighbors["neighbor_name"] == "")]
    df_neighbors = df_neighbors[~(df_neighbors["neighbor_name"].isna())]
    df_neighbors = df_neighbors[["neighbor_name", "cluster_key"]]
    df_neighbors.columns = ["site_id", "cluster_key"]

    df = pd.concat([df_neighbors, df_list_of_upgrades[["site_id", "cluster_key"]]]).drop_duplicates()

    df_list_of_upgrades.rename(columns={"selected_band": "bands_upgraded"}, inplace=True)
    df_neighbors_of_upgrades = df.merge(df_list_of_upgrades[["cluster_key", "bands_upgraded", "week_of_the_upgrade"]],
                                        on="cluster_key", how="left")

    print("df_neighbors_of_upgrades columns",df_neighbors_of_upgrades.columns)
    print("df_neighbors_of_upgrades shape",df_neighbors_of_upgrades.shape)

    df_neighbors_of_upgrades.to_parquet(neighbors_of_upgrades_data_output.path)
