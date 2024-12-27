from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_sites_to_remove_data(project_id: str,
                              location: str,
                              table_id: str,
                              query_results_data_output: Output[Dataset]):
    """Load sites to remove data from BigQuery and save the results as a parquet file for vertex pipeline.

    Args:
        project_id (str): GCP project ID.
        location (str): GCP location.
        table_id (str): BigQuery table ID.
        query_results_data_output (Output[Dataset]): Output dataset to store query results.

    Returns:
        None
    """
    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT *
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
