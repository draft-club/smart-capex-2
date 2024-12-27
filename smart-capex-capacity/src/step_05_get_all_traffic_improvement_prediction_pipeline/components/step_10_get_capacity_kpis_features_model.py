from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def get_capacity_kpis_features_model(operation_to_aggregate_cells:str,
                                     cell_affected_data_input: Input[Dataset],
                                     capacity_kpis_features_data_output:Output[Dataset]):
    """Compute capacity KPIs features and save the results as a parquet file for vertex pipeline.

    Args:
        operation_to_aggregate_cells (str): Operation to aggregate cells (e.g., 'mean', 'sum').
        cell_affected_data_input (Input[Dataset]): Input dataset containing cell affected data.
        capacity_kpis_features_data_output (Output[Dataset]): Output dataset to store capacity KPIs features.
    """
    import pandas as pd

    df_cell_affected = pd.read_parquet(cell_affected_data_input.path)

    capacity_kpis_features = ["average_number_of_users_in_queue", "average_throughput_user_dl"]

    def get_lag_between_two_week_periods(week_period_1, week_period_2):
        """Calculate the lag between two week periods.

        Args:
            week_period_1 (str): First week period.
            week_period_2 (str): Second week period.

        Returns:
            int: Lag between the two week periods.

        """
        week_period_1, week_period_2 = str(int(float(week_period_1))), str(int(float(week_period_2)))

        year1 = int(week_period_1[:4])
        week1 = int(week_period_1[-2:])
        year2 = int(week_period_2[:4])
        week2 = int(week_period_2[-2:])

        return - (53 * year1 + week1 - (53 * year2 + week2))


    def get_capacity_kpi_before(df_capacity_kpis, capacity_kpis_features, operation_to_aggregate_cells):
        """Compute capacity KPIs before the upgrade.

        Args:
            df_capacity_kpis (pd.DataFrame): DataFrame containing capacity KPIs.
            capacity_kpis_features (list): List of capacity KPIs features.
            operation_to_aggregate_cells (str): Operation to aggregate cells.

        Returns:
            pd.DataFrame: DataFrame containing aggregated capacity KPIs before the upgrade.
        """

        df_capacity_kpis["week_of_the_upgrade"] = df_capacity_kpis["week_of_the_upgrade"].apply(int).apply(str)
        df_capacity_kpis["week_period"] = df_capacity_kpis["week_period"].apply(int).apply(str)

        df_capacity_kpis["lag_between_upgrade"] = df_capacity_kpis[["week_of_the_upgrade", "week_period"]].apply(
                                                lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

        df_lag_between_upgrade_range = df_capacity_kpis[(df_capacity_kpis["lag_between_upgrade"] < 0)
                                                        & (df_capacity_kpis["lag_between_upgrade"] >= -8)]

        df_before = (df_lag_between_upgrade_range[capacity_kpis_features].agg({operation_to_aggregate_cells}).reset_index())

        return df_before

    df_cell_affected_grouped_columns = ["site_id", "week_date", "week_period", "cell_tech",
                                        "week_of_the_upgrade", "bands_upgraded", "tech_upgraded"]

    df_capacity_kpis_features = (df_cell_affected.groupby(df_cell_affected_grouped_columns)[capacity_kpis_features]
                                 .agg(operation_to_aggregate_cells).reset_index())

    df_capacity_kpis_features_grouped_columns = ["site_id", "cell_tech", "week_of_the_upgrade",
                                                 "bands_upgraded", "tech_upgraded"]

    df_capacity_kpis_features = df_capacity_kpis_features.groupby(df_capacity_kpis_features_grouped_columns).apply(
                                                                get_capacity_kpi_before,
                                                                capacity_kpis_features=capacity_kpis_features,
                                                                operation_to_aggregate_cells=operation_to_aggregate_cells
                                                                ).reset_index()

    df_capacity_kpis_features = df_capacity_kpis_features[["site_id", "cell_tech"] + capacity_kpis_features]

    df_capacity_kpis_features_pivoted = pd.pivot_table(df_capacity_kpis_features,
                                                       values=capacity_kpis_features,
                                                       index=["site_id"], columns=["cell_tech"],
                                                       aggfunc=operation_to_aggregate_cells, fill_value=None).reset_index()

    df_capacity_kpis_features_pivoted.columns = ["_".join(col).strip() for col
                                                 in df_capacity_kpis_features_pivoted.columns.values]

    df_capacity_kpis_features_pivoted.rename(columns={"site_id_": "site_id"}, inplace=True)

    df_capacity_kpis_features_pivoted.to_parquet(capacity_kpis_features_data_output.path)
