from kfp.dsl import Dataset, Input, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_increase_of_arpu_by_the_upgrade(increase_of_arpu_by_the_upgrade_table_id: str,
                                            project_id: str,
                                            location: str,
                                            predicted_increase_in_traffic_by_the_upgrade_data_input: Input[Dataset]):
    """
    Compute the increase of ARPU due to the upgrade and upload the results to a BigQuery table for other Vertex pipelines.

    Args:
        increase_of_arpu_by_the_upgrade_table_id (str): It holds BigQuery table ID where the results will be stored.
        project_id (str): It holds GCP project ID.
        location (str): It holds the location of the BigQuery tables.
        predicted_increase_in_traffic_by_the_upgrade_data_input (Input[Dataset]): It holds input dataset containing 
                                                                    predicted increase in traffic by the upgrade DataFrame.
    """
    import pandas as pd
    import pandas_gbq

    df_increase_arpu = pd.read_parquet(predicted_increase_in_traffic_by_the_upgrade_data_input.path)

    df_increase_arpu = df_increase_arpu[
        ['site_id', 'site_area', 'bands_upgraded', 'week_date', 'traffic_before_voice',
         'increase_of_traffic_after_the_upgrade_voice',
         'total_traffic_to_compute_increase_voice', 'increase_voice',
         'week_of_the_upgrade', 'year', 'week', 'week_period',
         'lag_between_upgrade', 'total_increase_voice',
         'traffic_increase_due_to_the_upgrade_voice',
         'traffic_before_data', 'increase_of_traffic_after_the_upgrade_data',
         'total_traffic_to_compute_increase_data', 'increase_data',
         'total_increase_data', 'traffic_increase_due_to_the_upgrade_data',
         'unit_price_data_mobile', 'unit_price_voice_min',
         'unit_price_data_mobile_with_the_decrease',
         'unit_price_voice_with_the_decrease']]

    df_increase_arpu['arpu_increase_due_to_the_upgrade_data'] = df_increase_arpu['traffic_increase_due_to_the_upgrade_data']
    df_increase_arpu['arpu_increase_due_to_the_upgrade_voice'] = \
                                                        df_increase_arpu['traffic_increase_due_to_the_upgrade_voice'] * \
                                                        1000 * 60 * df_increase_arpu['unit_price_voice_with_the_decrease']

    df_increase_arpu = df_increase_arpu.drop_duplicates(keep="first")

    pandas_gbq.to_gbq(df_increase_arpu,
                      increase_of_arpu_by_the_upgrade_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists='replace')
