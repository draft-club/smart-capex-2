from kfp.dsl import (Dataset,
                     Input,
                     component)
from utils.config import pipeline_config

# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def get_cluster_of_affected_sites(project_id:str,
                                  location:str,
                                  sites_to_remove_table_id:str,
                                  list_of_upgrades_table_id:str,
                                  max_number_of_neighbour:int,
                                  cell_affected_data_input: Input[Dataset],
                                  sites_distances_data_input:Input[Dataset]):
    """Computes the cluster of affected sites of the upgrades

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        sites_to_remove_table_id (str): It holds the resource name for sites to remove on BigQuery 
        list_of_upgrades_table_id (str): It holds the resource name for list of upgrades on BigQuery 
        max_number_of_neighbour (int): It holds maximum number of neighbors to consider
        cell_affected_data_input (Input[Dataset]): It holds cell affected of the upgrade data
        sites_distances_data_input (Input[Dataset]): It holds holds distances between sites data
        sites_to_remove_data_output (Output[Dataset]): It holds sites to remove data
        list_of_upgrades_data_output (Output[Dataset]): It holds list of upgrades data

    Returns:
        sites_to_remove_data_output (Output[Dataset]): It holds sites to remove data
        list_of_upgrades_data_output (Output[Dataset]): It holds list of upgrades data
    """

    import pandas_gbq
    import pandas as pd


    df_cell_affected = pd.read_parquet(cell_affected_data_input.path)
    df_distance = pd.read_parquet(sites_distances_data_input.path)

    def get_lag_between_two_week_periods(week_period_1, week_period_2):
        week_period_1, week_period_2 = str(int(float(week_period_1))), str(int(float(week_period_2)))
        year_1 = int(week_period_1[:4])
        week_1 = int(week_period_1[-2:])
        year_2 = int(week_period_2[:4])
        week_2 = int(week_period_2[-2:])
        return - (53 * year_1 + week_1 - (53 * year_2 + week_2))

    def compute_neighbour_site(site_id, df_distance, position):
        try:
            distance_vector = df_distance[site_id]
        except (KeyError, TypeError, AttributeError):
            return ""

        distance = 3.5
        distance_vector = distance_vector[(distance_vector > 0) & (distance_vector < distance)].sort_values()
        if position > len(distance_vector):
            return ""
        # else:
        return distance_vector.index[position - 1]

    def compute_list_neighbours_sites(site_id, bands_upgraded, df_distance):
        max_number = 2
        try:
            distance_vector = df_distance[site_id]
        except (KeyError, TypeError):
            return None

        if "L800" in bands_upgraded or "U900" in bands_upgraded:
            distance = 3.5
        else:
            distance = 2
        distance_vector = distance_vector[(distance_vector > 0) & (distance_vector < distance)]
        return distance_vector.sort_values()[0:max_number].index

    df_cell_affected_subset= df_cell_affected[["site_id", "week_of_the_upgrade", "bands_upgraded", "tech_upgraded"]]

    df_list_of_upgrades = df_cell_affected_subset.drop_duplicates()

    df_list_of_upgrades["cluster_key"] = df_list_of_upgrades \
                                      .apply(lambda x:"_".join([str(x["site_id"]), str(x["week_of_the_upgrade"])]), axis=1)

    for i in range(0, max_number_of_neighbour):
        df_list_of_upgrades["neighbor_" + str(i + 1)] = df_list_of_upgrades[["site_id", "bands_upgraded"]] \
                                            .apply(lambda x:compute_neighbour_site(x.iloc[0], df_distance, i + 1))
        df_list_of_upgrades.fillna("", inplace=True)

        df_list_of_upgrades["cluster_key"] = (df_list_of_upgrades["cluster_key"].astype(str) +
                                              "_" +
                                              df_list_of_upgrades["neighbor_" + str(i + 1)])

    neighbors = {}
    sites_to_remove = []
    for index, row in df_list_of_upgrades.iterrows():
        neighbors_sites = compute_list_neighbours_sites(row[0], row[2], df_distance)
        print(index)
        if neighbors_sites is None:
            row["neighbors"] = [""]
            neighbors[row[0]] = []
            continue

        neighbors[row[0]] = neighbors_sites.values
        ## Compute if there is an upgrade on sites on the same cluster in a period less than
        # 2 months
        for site in neighbors_sites:
            if site in df_list_of_upgrades["site_id"].values:
                upgrade = df_list_of_upgrades[df_list_of_upgrades["site_id"] == site]
                upgrade_week = upgrade["week_of_the_upgrade"].values[0]

                if abs(get_lag_between_two_week_periods(str(row[1]), str(upgrade_week))) <= 8:
                    if row["site_id"] not in sites_to_remove:
                        sites_to_remove.append(row["site_id"])

        df_sites_to_remove = pd.DataFrame({"site_id": sites_to_remove})

    pandas_gbq.to_gbq(df_sites_to_remove, sites_to_remove_table_id,
                      project_id=project_id, location=location, if_exists='replace')
    pandas_gbq.to_gbq(df_list_of_upgrades, list_of_upgrades_table_id, project_id=project_id,
                      location=location, if_exists='replace')
