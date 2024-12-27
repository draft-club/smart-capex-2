from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_list_of_upgrades_data(project_id: str,
                               location: str,
                               table_id: str,
                               query_results_data_output: Output[Dataset]):
    """Load list of upgrades data from BigQuery and save the results as a parquet file for vertex pipeline.

    Args:
        project_id (str): GCP project ID.
        location (str): GCP location.
        table_id (str): BigQuery table ID.
        query_results_data_output (Output[Dataset]): Output dataset to store query results.
    """

    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT cluster_key, week_of_the_upgrade, site_id, tech_upgraded, bands_upgraded
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
