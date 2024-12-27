from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_affected_cells_data(project_id: str,
                             location: str,
                             table_id: str,
                             query_results_data_output: Output[Dataset]):
    """Load affected cells data from BigQuery and save the results as a parquet file for vertex pipeline.

    Args:
        project_id (str): GCP project ID.
        location (str): GCP location.
        table_id (str): BigQuery table ID.
        query_results_data_output (Output[Dataset]): Output dataset to store query results.
    """

    # imports
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT site_id, week_date, week_period, cell_tech, 
                   week_of_the_upgrade, bands_upgraded, tech_upgraded, average_throughput_user_dl, average_number_of_users_in_queue
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    df_query["week_date"] = pd.to_datetime(df_query["week_date"]).dt.date

    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
