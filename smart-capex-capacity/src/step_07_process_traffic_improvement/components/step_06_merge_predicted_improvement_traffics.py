from kfp.dsl import (Dataset,
                     Input,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def merge_predicted_improvement_traffics(project_id: str,
                                         location: str,
                                         predicted_increase_in_traffic_by_the_upgrade_table_id: str,
                                         data_predicted_kpis_trend_data_input: Input[Dataset],
                                         voice_predicted_kpis_trend_data_input: Input[Dataset]):
    """Merges the predicted improvement in traffic KPIs for data and voice after an upgrade.

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        predicted_increase_in_traffic_by_the_upgrade_table_id (str): It holds the resource name on BigQuery
        data_predicted_kpis_trend_data_input (Input[Dataset]):  It holds predicted data for data KPIs.
        voice_predicted_kpis_trend_data_input (Input[Dataset]): It holds predicted data for voice KPIs.

    Returns:
        None
    """

    import pandas as pd
    import pandas_gbq

    df_predicted_increase_in_traffic_data_by_the_upgrade = pd.read_parquet(data_predicted_kpis_trend_data_input.path)
    df_predicted_increase_in_traffic_voice_by_the_upgrade = pd.read_parquet(voice_predicted_kpis_trend_data_input.path)

    df_predicted_increase_in_traffic_data_by_the_upgrade = (df_predicted_increase_in_traffic_data_by_the_upgrade
        .rename(columns={
        'date': 'week_date',
        'traffic_before': 'traffic_before_data',
        'increase_of_traffic_after_the_upgrade': 'increase_of_traffic_after_the_upgrade_data',
        'total_traffic_to_compute_increase': 'total_traffic_to_compute_increase_data',
        'increase': 'increase_data',
        'total_increase': 'total_increase_data',
        'traffic_increase_due_to_the_upgrade': 'traffic_increase_due_to_the_upgrade_data'}))

    df_predicted_increase_in_traffic_voice_by_the_upgrade = (df_predicted_increase_in_traffic_voice_by_the_upgrade
        .rename(columns={
        'date': 'week_date',
        'traffic_before': 'traffic_before_voice',
        'increase_of_traffic_after_the_upgrade': 'increase_of_traffic_after_the_upgrade_voice',
        'total_traffic_to_compute_increase': 'total_traffic_to_compute_increase_voice',
        'increase': 'increase_voice',
        'total_increase': 'total_increase_voice',
        'traffic_increase_due_to_the_upgrade': 'traffic_increase_due_to_the_upgrade_voice'}))

    df_predicted_increase_in_traffic_by_the_upgrade = df_predicted_increase_in_traffic_voice_by_the_upgrade.merge(
                                                    df_predicted_increase_in_traffic_data_by_the_upgrade,
                                                    on=["site_id", "bands_upgraded", "site_area", "week_date",
                                                        "week_of_the_upgrade","year", "week",
                                                        "week_period", "lag_between_upgrade"],
                                                    how="inner")
    ### HINT: remove columns that startwith "level"
    drop_columns = df_predicted_increase_in_traffic_by_the_upgrade.columns.str.startswith('level')
    df_predicted_increase_in_traffic_by_the_upgrade = df_predicted_increase_in_traffic_by_the_upgrade.loc[:, ~drop_columns]

    pandas_gbq.to_gbq(df_predicted_increase_in_traffic_by_the_upgrade, predicted_increase_in_traffic_by_the_upgrade_table_id,
                      project_id=project_id, location=location, if_exists='replace')
