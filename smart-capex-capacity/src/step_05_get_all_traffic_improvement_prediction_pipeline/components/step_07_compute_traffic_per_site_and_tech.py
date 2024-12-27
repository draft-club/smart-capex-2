from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_traffic_per_site_and_tech(kpi_to_compute_upgrade_effect:str,
                                      traffic_weekly_kpis_data_input: Input[Dataset],
                                      neighbors_of_upgrades_data_input: Input[Dataset],
                                      traffic_weekly_kpis_site_data_output: Output[Dataset],
                                      traffic_weekly_kpis_site_tech_data_output: Output[Dataset]):
    """Computes traffic per site and technology and saves the results as parquet files for vertex pipeline.

    Args:
        kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the input dataset containing traffic weekly KPIs dataframe.
        neighbors_of_upgrades_data_input (Input[Dataset]): It holds the input dataset containing
                                                            neighbors of upgrades dataframe.
        traffic_weekly_kpis_site_data_output (Output[Dataset]): It holds the output dataset to store
                                                                traffic weekly KPIs per site dataframe.
        traffic_weekly_kpis_site_tech_data_output (Output[Dataset]): It holds the output dataset to store traffic weekly
                                                                        KPIs per site and technology dataframe.

    """

    import pandas as pd

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    df_neighbors_of_upgrades = pd.read_parquet(neighbors_of_upgrades_data_input.path)

    df_traffic_weekly_kpis_site = (df_traffic_weekly_kpis
                                   .groupby(["site_id", "week_date", "week_period"])
                                   [kpi_to_compute_upgrade_effect].sum().reset_index())

    # traffic per site and cell tech
    df_traffic_weekly_kpis_site_tech = (df_traffic_weekly_kpis
                                        .groupby(["site_id", "cell_tech", "week_date", "week_period"])
                                        [kpi_to_compute_upgrade_effect].sum().reset_index())
    del df_traffic_weekly_kpis
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(df_neighbors_of_upgrades, on="site_id", how="left")
    df_traffic_weekly_kpis_site_tech = df_traffic_weekly_kpis_site_tech.merge(df_neighbors_of_upgrades,
                                                                              on="site_id", how="left")

    df_traffic_weekly_kpis_site.to_parquet(traffic_weekly_kpis_site_data_output.path)
    df_traffic_weekly_kpis_site_tech.to_parquet(traffic_weekly_kpis_site_tech_data_output.path)
