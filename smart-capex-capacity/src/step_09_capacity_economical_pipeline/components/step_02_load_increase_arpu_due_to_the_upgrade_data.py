from kfp.dsl import (Dataset,
                     Output,
                     component)
from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def load_increase_arpu_due_to_the_upgrade_data(project_id: str,
                                               location: str,
                                               table_id: str,
                                               query_results_data_output: Output[Dataset]):
    """It loads the increase arpu due to the upgrade data from GCP

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        table_id (str): It holds the resource name on BigQuery
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery

    Returns:
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery
    """

    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    # week_date instead of date
    query_statement = f"""SELECT site_id, bands_upgraded, "site_area", year, week_date,
    arpu_increase_due_to_the_upgrade_data, arpu_increase_due_to_the_upgrade_voice
                          FROM {table_id}"""

    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
