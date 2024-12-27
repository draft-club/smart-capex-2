from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_data(project_id: str,
              location: str,
              table_id: str,
              query_results_data_output: Output[Dataset]):
    """Load the data from GCP BigQuery table.

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        table_id (str): It holds the resource name on BigQuery
        query_results_data_output (Output[Dataset]): It holds the processed data returned from BigQuery

    Returns 
        query_results_data_output (Output[Dataset]): It holds the processed data returned from BigQuery
    """

    # imports
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
        SELECT *
        FROM {table_id}
    """

    df_query_result = client.query(query_statement).to_dataframe()
    print("Query DataFrame shape: ", df_query_result.shape)

    # extract date columns with type `dbdate`
    date_columns = df_query_result.dtypes[df_query_result.dtypes == "dbdate"].index
    # convert `dbdate` columns type to date object
    df_query_result[date_columns] = df_query_result[date_columns].apply(lambda column: pd.to_datetime(column).dt.date)

    df_query_result.to_parquet(query_results_data_output.path)
