from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_traffic_model_features(compute_target: bool,
                                   weeks_to_wait_after_upgrade: int,
                                   weeks_to_wait_after_upgrade_max: int,
                                   kpi_to_compute_upgrade_effect: str,
                                   traffic_weekly_kpis_cluster_data_input: Input[Dataset],
                                   traffic_weekly_kpis_cluster_tech_data_input: Input[Dataset],
                                   traffic_features_data_output: Output[Dataset],
                                   traffic_site_tech_data_output: Output[Dataset]):
    """Computes traffic model features and saves the results as parquet files.

    Args:
        compute_target (bool): It holds the flag to compute the target variable.
        weeks_to_wait_after_upgrade (int): It holds the number of weeks to wait after the upgrade.
        weeks_to_wait_after_upgrade_max (int): It holds the maximum number of weeks to wait after the upgrade.
        kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.
        traffic_weekly_kpis_cluster_data_input (Input[Dataset]): It holds the input dataset containing
                                                                    traffic weekly KPIs per cluster.
        traffic_weekly_kpis_cluster_tech_data_input (Input[Dataset]): It holds the input dataset containing 
                                                                        traffic weekly KPIs per cluster and technology.
        traffic_features_data_output (Output[Dataset]): It holds the output dataset to store traffic features.
        traffic_site_tech_data_output (Output[Dataset]): It holds the output dataset to store traffic site technology data.
    """
    import numpy as np
    import pandas as pd

    df_traffic_weekly_kpis_cluster = pd.read_parquet(traffic_weekly_kpis_cluster_data_input.path)
    df_traffic_weekly_kpis_cluster_tech = pd.read_parquet(traffic_weekly_kpis_cluster_tech_data_input.path)

    print(df_traffic_weekly_kpis_cluster.columns)
    print(df_traffic_weekly_kpis_cluster_tech.columns)


    def get_lag_between_two_week_periods(week_period_1, week_period_2):
        """Computes the lag between two week periods.

        Args:
            week_period_1 (str): It holds the first week period.
            week_period_2 (str): It holds the second week period.

        Returns:
            int: The lag between the two week periods.
        """

        week_period_1, week_period_2 = str(int(float(week_period_1))), str(int(float(week_period_2)))
        year_1 = int(week_period_1[:4])
        week_1 = int(week_period_1[-2:])
        year_2 = int(week_period_2[:4])
        week_2 = int(week_period_2[-2:])

        return - (53 * year_1 + week_1 - (53 * year_2 + week_2))

    def compute_lag_between_upgrade(df_cluster):
        """Computes the lag between upgrades for a cluster.

        Args:
            df_cluster (pd.DataFrame): It holds the DataFrame containing cluster data.

        Returns:
            pd.DataFrame: The DataFrame with the lag between upgrades.
        """

        df_cluster["week_of_the_upgrade"] = df_cluster["week_of_the_upgrade"].apply(int).apply(str)
        df_cluster["week_period"] = df_cluster["week_period"].apply(int).apply(str)
        df_cluster["lag_between_upgrade"] = df_cluster[["week_of_the_upgrade", "week_period"]].apply(
            lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)
        return df_cluster

    def compute_traffic_features_per_cluster(df_cluster,
                                             kpi_to_compute_upgrade_effect,
                                             compute_target=True):
        """Compute traffic features for a cluster.

        Args:
            df_cluster (pd.DataFrame): It holds the DataFrame containing cluster data.
            kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.
            compute_target (bool): It holds the flag to compute the target variable.

        Returns:
            pd.DataFrame: The DataFrame with computed traffic features.
        """

        df_final = df_cluster.reset_index(drop=True)[["cluster_key"]].drop_duplicates()
        df_cluster = compute_lag_between_upgrade(df_cluster)

        # mask defined in a separate line for pylint
        mask_lag_between_upgrade = (df_cluster["lag_between_upgrade"] < 0) & (df_cluster["lag_between_upgrade"] >= -8)
        df_lag_between_upgrade_range = df_cluster[mask_lag_between_upgrade]

        df_before = df_lag_between_upgrade_range[kpi_to_compute_upgrade_effect].agg(
                                                                {"mean", "std", "median", "min", "max"}).reset_index()

        df_before = pd.pivot_table(df_before, values=kpi_to_compute_upgrade_effect,
                                   columns=["index"], aggfunc=np.sum)

        df_before.columns = ["".join([kpi_to_compute_upgrade_effect, "_", i]) for i in df_before.columns]
        df_before.reset_index(drop=True, inplace=True)
        df_final = pd.concat([df_final, df_before], axis=1)

        if compute_target:
            # mask defined in a separate line for pylint
            mask_lag_between_upgrade = (df_cluster["lag_between_upgrade"] > weeks_to_wait_after_upgrade) \
                                        & (df_cluster["lag_between_upgrade"] <= weeks_to_wait_after_upgrade_max)

            df_target = df_cluster[mask_lag_between_upgrade][kpi_to_compute_upgrade_effect].mean()

            df_final["target_variable"] = df_target

        return df_final

    def compute_traffic_features_per_cluster_tech(df_cluster_tech, kpi_to_compute_upgrade_effect):
        """Computes traffic features for a cluster and technology.

        Args:
            df_cluster_tech (pd.DataFrame): It holds the DataFrame containing cluster and technology data.
            kpi_to_compute_upgrade_effect (str): It holds the KPI to compute the upgrade effect.

        Returns:
            pd.DataFrame: The DataFrame with computed traffic features for the cluster and technology.
        """

        df_final = df_cluster_tech.reset_index(drop=True)[["cluster_key", "cell_tech"]].drop_duplicates()
        df_cluster_tech = compute_lag_between_upgrade(df_cluster_tech)

        # defined mask in a separate line for pylint
        mask_lag_between_upgrades = (df_cluster_tech["lag_between_upgrade"] < 0) \
                                        & (df_cluster_tech["lag_between_upgrade"] >= -8)
        df_lag_between_upgrade_range = df_cluster_tech[mask_lag_between_upgrades]

        df_before = (df_lag_between_upgrade_range.groupby("cell_tech")[
                        kpi_to_compute_upgrade_effect].agg({"mean", "std", "median", "min", "max"}).reset_index())

        df_before.columns = [kpi_to_compute_upgrade_effect + "_tech_" + a if "cell_tech" not in a
                                else "tech_upgraded" for a in df_before.columns.to_flat_index()]

        df_before.reset_index(drop=True, inplace=True)

        df_final = pd.concat([df_final, df_before], axis=1)
        return df_final

    ### Compute the model features
    df_traffic_features = df_traffic_weekly_kpis_cluster.groupby("cluster_key").apply(
                                                                compute_traffic_features_per_cluster,
                                                                kpi_to_compute_upgrade_effect=kpi_to_compute_upgrade_effect,
                                                                compute_target=compute_target).reset_index(drop=True)

    # Compute site-tech kpi aggregate
    df_traffic_site_tech = (df_traffic_weekly_kpis_cluster_tech.groupby(["cluster_key", "cell_tech"]).apply(
                                        compute_traffic_features_per_cluster_tech,
                                        kpi_to_compute_upgrade_effect=kpi_to_compute_upgrade_effect).reset_index(drop=True))

    if compute_target:
        df_traffic_features = df_traffic_features[~df_traffic_features["target_variable"].isna()]

    df_traffic_features.to_parquet(traffic_features_data_output.path)
    df_traffic_site_tech.to_parquet(traffic_site_tech_data_output.path)
