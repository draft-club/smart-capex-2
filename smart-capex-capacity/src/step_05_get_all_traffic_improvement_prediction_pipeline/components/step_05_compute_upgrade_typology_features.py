from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_upgrade_typology_features(bands_to_consider:list,
                                      max_number_of_neighbors:int,
                                      traffic_weekly_kpis_data_input: Input[Dataset],
                                      list_of_upgrades_data_input:Input[Dataset],
                                      upgrades_features_data_output: Output[Dataset]):
    """Compute upgrade typology features and save the results as a parquet file for vertex pipeline.

    Args:
        bands_to_consider (list): It holds list of bands to consider.
        max_number_of_neighbors (int): It holds the maximum number of neighbors.
        traffic_weekly_kpis_data_input (Input[Dataset]): Input dataset containing traffic weekly KPIs.
        list_of_upgrades_data_input (Input[Dataset]): Input dataset containing list of upgrades.
        upgrades_features_data_output (Output[Dataset]): Output dataset to store the computed upgrade features.
    """

    import pandas as pd
    import numpy as np

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    df_list_of_upgrades = pd.read_parquet(list_of_upgrades_data_input.path)

    def compute_sites_attributes(df, df_traffic_weekly_kpis):
        """Compute site attributes for a given upgrades cluster by merging the upgrade features
            with the traffic weekly KPIs dataframes and computing the pivot table of the resulted dataframe 
            to give the KPIs summation of each cluster.

        Args:
            df (pd.DataFrame): DataFrame containing cluster data.
            df_traffic_weekly_kpis (pd.DataFrame): DataFrame containing traffic weekly KPIs.

        Returns:
            pd.DataFrame: DataFrame with computed site attributes.
        """

        df = df.reset_index()
        df_weekly = df_traffic_weekly_kpis.merge(df, on="site_id", how="inner")

        df_weekly = df_weekly[df_weekly["cell_band"].isin(bands_to_consider)]

        df_weekly["week_period"] = df_weekly["week_period"].apply(str)
        df_weekly["week_of_the_upgrade"] = df_weekly["week_of_the_upgrade"].apply(str)

        df_weekly = df_weekly[df_weekly["week_period"] < df_weekly["week_of_the_upgrade"]]
        df_weekly = df_weekly[["site_id", "cluster_key", "cell_band", "cell_tech"]].drop_duplicates()

        df_weekly["type"] = "site"

        df_number_tech = df_weekly.groupby(["cluster_key", "cell_tech", "type"])["site_id"].count().reset_index()

        df_number_tech = pd.pivot_table(df_number_tech,
                                        values="site_id",
                                        index=["cluster_key"],
                                        columns=["cell_tech", "type"],
                                        fill_value=0,
                                        aggfunc=np.sum)

        df_number_tech.columns = ["_".join(col).strip() for col in df_number_tech.columns.values]

        df_number_tech = df_number_tech.reset_index()
        ## Number of sites on the cluster
        df_number_tech["number_of_sites_on_the_cluster"] = df.shape[0] - 1
        return df_number_tech


    list_neighbors = ["neighbor" + "_" + str(i + 1) for i in range(max_number_of_neighbors)]

    df_list_of_upgrades_neighbors= df_list_of_upgrades[["cluster_key", "week_of_the_upgrade"] + list_neighbors]

    df_neighbors = pd.melt(df_list_of_upgrades_neighbors,
                           id_vars=["cluster_key", "week_of_the_upgrade"],
                           value_vars=list_neighbors)

    df_neighbors.columns = ["cluster_key", "week_of_the_upgrade", "variable", "site_id"]
    df_neighbors.drop(columns=["variable"], inplace=True)
    df_neighbors = df_neighbors[~(df_neighbors["site_id"] == "")]
    df_neighbors.dropna(subset=["site_id"], inplace=True)

    df_list_of_upgrades = df_list_of_upgrades[["site_id", "cluster_key","week_of_the_upgrade"]].drop_duplicates()
    df_neighbors = df_neighbors.drop_duplicates()

    df_upgrades = pd.concat([df_neighbors, df_list_of_upgrades])

    df_upgrades_features = (df_upgrades.groupby("cluster_key")
                            .apply(compute_sites_attributes, df_traffic_weekly_kpis=df_traffic_weekly_kpis)
                            .reset_index(drop=True))

    df_upgrades_features.fillna(0, inplace=True)

    df_upgrades_features.to_parquet(upgrades_features_data_output.path)
