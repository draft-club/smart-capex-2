from kfp.dsl import (Dataset,
                     Output,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_revenues_per_unit_traffic_data(project_id: str,
                                        location: str,
                                        table_id: str,
                                        query_results_data_output: Output[Dataset]):
    """It loads the revenues per unit traffic data from GCP

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        table_id (str): It holds the resource name on BigQuery
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery

    Returns:
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery
    """

    # Imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    # revenue_total_mobile instead of revenue_total_site
    # opex_clients_commissions instead of opex_clients
    query_statement = f"""SELECT site_id, revenue_total_mobile, revenue_data_mobile,
    revenue_voice_mobile, opex_clients_commissions FROM {table_id}"""
    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
