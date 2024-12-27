from kfp.dsl import (Dataset,
                     Output,
                     component)
from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def load_capex_data(project_id: str,
                   location: str,
                   table_id: str,
                   query_results_data_output: Output[Dataset]):
    """It loads the capex data from GCP

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

    query_statement = f""" SELECT bands, capex_cost
            FROM {table_id}
        """
    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
