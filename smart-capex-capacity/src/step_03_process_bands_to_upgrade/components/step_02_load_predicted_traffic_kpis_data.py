from kfp.dsl import (Dataset, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_predicted_traffic_kpis_data(project_id: str,
                                     location: str,
                                     table_id: str,
                                     query_results_data_output: Output[Dataset]):
    """It loads the predicted traffic KPIs data from GCP

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        table_id (str): It holds the resource name on BigQuery
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery

    Returns:
        query_results_data_output (Output[Dataset]): It holds the predicted traffic KPIs data returned from BigQuery
    """

    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT cell_name, week_date, cell_tech, cell_band, site_id, year, week, week_period,
            total_voice_traffic_kerlands, total_data_traffic_dl_gb, average_number_of_users_in_queue, average_throughput_user_dl
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
