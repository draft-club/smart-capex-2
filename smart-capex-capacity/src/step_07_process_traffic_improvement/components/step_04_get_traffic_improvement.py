from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def get_traffic_improvement(kpi_to_compute_upgrade_effect: str,
                            traffic_weekly_kpis_data_input: Input[Dataset],
                            selected_band_per_site_data_input: Input[Dataset],
                            traffic_features_future_upgrades_data_input: Input[Dataset],
                            traffic_weekly_kpis_site_data_output: Output[Dataset]):

    """Computes the traffic improvement after upgrades and filters the weekly KPIs by site.

    Args:
        kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the  traffic weekly KPIs data.
        selected_band_per_site_data_input (Input[Dataset]): It holds the selected band per site data.
        traffic_features_future_upgrades_data_input (Input[Dataset]): It holds the predictions and the increase of traffic.
        traffic_weekly_kpis_site_data_output (Output[Dataset]): It holds the filtered weekly KPIs by site.

    Returns:
        traffic_weekly_kpis_site_data_output (Output[Dataset]): It holds the filtered weekly KPIs by site.
    """

    import pandas as pd

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    df_selected_band_per_site = pd.read_parquet(selected_band_per_site_data_input.path)
    df_traffic_features_future_upgrades = pd.read_parquet(traffic_features_future_upgrades_data_input.path)

    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis.groupby(
        ["week_date", "week_period", "site_id"])[kpi_to_compute_upgrade_effect].sum().reset_index()

    df_selected_band_per_site_subset = df_selected_band_per_site[['site_id', 'congestion', 'tech_upgraded',
                                                                  'bands_upgraded', 'week_of_the_upgrade']]

    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(
        df_selected_band_per_site_subset, on="site_id", how="inner")

    # Merge with df_traffic_features_future_upgrades to bring back the increase improvment
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(
        df_traffic_features_future_upgrades[["site_id", "bands_upgraded", "increase_of_traffic_after_the_upgrade"]],
        on=["site_id", "bands_upgraded"], how="left")

    ## If the increase in negative, replace by 0
    df_traffic_weekly_kpis_site["increase_of_traffic_after_the_upgrade"] = df_traffic_weekly_kpis_site[
                                                                    "increase_of_traffic_after_the_upgrade"].clip(lower=0)

    ## Remove dismantled sites
    df_remove_sites = df_traffic_weekly_kpis.groupby(["site_id"])[kpi_to_compute_upgrade_effect].sum().reset_index()
    df_remove_sites = df_remove_sites.loc[df_remove_sites[kpi_to_compute_upgrade_effect] == 0]
    df_traffic_weekly_kpis_site_to_remove = df_traffic_weekly_kpis_site["site_id"].isin(df_remove_sites["site_id"].values)
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[~df_traffic_weekly_kpis_site_to_remove]

    df_remove_sites = df_traffic_weekly_kpis_site.groupby(["site_id"])[kpi_to_compute_upgrade_effect].count().reset_index()
    df_remove_sites = df_remove_sites.loc[df_remove_sites[kpi_to_compute_upgrade_effect] < 10]
    df_traffic_weekly_kpis_site_to_remove = df_traffic_weekly_kpis_site["site_id"].isin(df_remove_sites["site_id"].values)
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[~df_traffic_weekly_kpis_site_to_remove]

    ## Remove those sites with target variable equal to na
    df_traffic_weekly_kpis_site_nans = df_traffic_weekly_kpis_site["increase_of_traffic_after_the_upgrade"].isna()
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[~df_traffic_weekly_kpis_site_nans]

    df_traffic_weekly_kpis_site.to_parquet(traffic_weekly_kpis_site_data_output.path)
