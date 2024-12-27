from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_month_period(predicted_increase_in_traffic_by_the_upgrade_data_input: Input[Dataset],
                         predicted_increase_in_traffic_by_the_upgrade_data_output: Output[Dataset]):
    """
    Compute the month period from the given weekly traffic data and save the result DataFrame for Vertex pipeline.

    Args:
        predicted_increase_in_traffic_by_the_upgrade_data_input (Input[Dataset]): It holds input dataset containing
                                                                    predicted increase in traffic by the upgrade DataFrame.
        predicted_increase_in_traffic_by_the_upgrade_data_output (Output[Dataset]): Output dataset to store the computed
                                                                                        month period DataFrame.
    """
    import pandas as pd

    df_predicted_increase_in_traffic_by_the_upgrade = pd.read_parquet(
                                                            predicted_increase_in_traffic_by_the_upgrade_data_input.path)

    df_predicted_increase_in_traffic_by_the_upgrade['week_date'] = pd.to_datetime(
                                                            df_predicted_increase_in_traffic_by_the_upgrade['week_date'],
                                                            format='%Y-%m-%d')

    df_predicted_increase_in_traffic_by_the_upgrade['month'] = \
                                df_predicted_increase_in_traffic_by_the_upgrade["week_date"].dt.month

    df_predicted_increase_in_traffic_by_the_upgrade['month'] = \
                                df_predicted_increase_in_traffic_by_the_upgrade['month'].apply(lambda x: str(x).zfill(2))

    df_predicted_increase_in_traffic_by_the_upgrade['year'] = \
                                df_predicted_increase_in_traffic_by_the_upgrade['year'].apply(str)

    df_predicted_increase_in_traffic_by_the_upgrade['month_period'] = \
                                                                df_predicted_increase_in_traffic_by_the_upgrade['year'] + \
                                                                df_predicted_increase_in_traffic_by_the_upgrade['month']

    df_predicted_increase_in_traffic_by_the_upgrade.to_parquet(predicted_increase_in_traffic_by_the_upgrade_data_output.path)
