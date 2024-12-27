from kfp.dsl import (Dataset, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_b32_data(project_id: str,
                  location: str,
                  table_id: str,
                  query_results_data_output: Output[Dataset]):
    """It loads the b32 data from GCP

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        table_id (str): It holds the resource name on BigQuery
        query_results_data_output (Output[Dataset]): It holds the data returned from BigQuery

    Returns:
        query_results_data_output (Output[Dataset]): It holds the b32 data returned from BigQuery
    """

    # imports
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f""" SELECT file_date, site_id, `4g_devices`, band_32_devices
            FROM {table_id}
        """
    df_query = client.query(query_statement).to_dataframe()
    df_query["file_date"] = pd.to_datetime(df_query["file_date"]).dt.date
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
