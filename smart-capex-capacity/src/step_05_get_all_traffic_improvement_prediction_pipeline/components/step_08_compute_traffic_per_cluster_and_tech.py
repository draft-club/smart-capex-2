from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_traffic_per_cluster_and_tech(kpi_to_compute_upgrade_effect:str,
                                         traffic_weekly_kpis_site_data_input:Input[Dataset],
                                         traffic_weekly_kpis_site_tech_data_input:Input[Dataset],
                                         traffic_weekly_kpis_cluster_data_output:Output[Dataset],
                                         traffic_weekly_kpis_cluster_tech_data_output:Output[Dataset]):
    """Computes traffic per site and technology and saves the results as parquet files for vertex pipeline.

    Args:
        kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the input dataset containing traffic weekly KPIs.
        neighbors_of_upgrades_data_input (Input[Dataset]): It holds the input dataset containing neighbors of upgrades.
        traffic_weekly_kpis_site_data_output (Output[Dataset]): It holds the output dataset to store
                                                                traffic weekly KPIs per site.
        traffic_weekly_kpis_site_tech_data_output (Output[Dataset]): It holds the output dataset to store traffic weekly
                                                                        KPIs per site and technology.
    """

    import pandas as pd

    df_traffic_weekly_kpis_site = pd.read_parquet(traffic_weekly_kpis_site_data_input.path)
    df_traffic_weekly_kpis_site_tech =  pd.read_parquet(traffic_weekly_kpis_site_tech_data_input.path)


    df_traffic_weekly_kpis_cluster = df_traffic_weekly_kpis_site.groupby(
                                    ["cluster_key", "week_date", "week_period", "week_of_the_upgrade", "bands_upgraded"])[
                                                kpi_to_compute_upgrade_effect].sum().reset_index()

    ## traffic per cluster keybands_upgrade and week of the upgrade and cell tech
    df_traffic_weekly_kpis_cluster_tech = df_traffic_weekly_kpis_site_tech.groupby(
                            ["cluster_key", "cell_tech", "week_date", "week_period",
                                "week_of_the_upgrade", "bands_upgraded"])[kpi_to_compute_upgrade_effect].sum().reset_index()


    df_traffic_weekly_kpis_cluster.to_parquet(traffic_weekly_kpis_cluster_data_output.path)
    df_traffic_weekly_kpis_cluster_tech.to_parquet(traffic_weekly_kpis_cluster_tech_data_output.path)
