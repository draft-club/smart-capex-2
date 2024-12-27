from kfp.dsl import (Dataset,
                     Output,
                     component)
from utils.config import pipeline_config

# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def load_sites_data(project_id: str,
                    location: str,
                    table_id: str,
                    query_results_data_output: Output[Dataset]):
    """It loads the data from GCP

    Args:
       project_id (str): It holds the project_id of GCP
       location (str): It holds the location assigned to the project on GCP
       table_id (str): It holds the resource name on BigQuery
       query_results_data_output (Output[Dataset]): It holds the processed sites returned from BigQuery
        
    Returns:
        query_results_data_output (Output[Dataset]): It holds the processed sites returned from BigQuery
    """
    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT longitude, latitude, site_id, cell_name, cell_tech, cell_band, region
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
