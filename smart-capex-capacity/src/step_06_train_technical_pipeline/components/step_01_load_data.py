from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_data(project_id: str,
              location: str,
              table_id: str,
              query_results_data_output: Output[Dataset]):
    """Load data from BigQuery and save the resulted dataframe as a parquet file for vertex pipeline.

    Args:
        project_id (str): It holds the GCP project ID.
        location (str): It holds the GCP project location.
        table_id (str): It holds the BigQuery table ID.
        query_results_data_output (Output[Dataset]): It holds the output dataset to store the query results.
    """

    # imports
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    query_statement = f"""
            SELECT *
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()

    # extract date columns with type `dbdate`
    date_columns = df_query.dtypes[df_query.dtypes == "dbdate"].index

    # convert `dbdate` columns type to date object
    df_query[date_columns] = df_query[date_columns].apply(lambda column: pd.to_datetime(column).dt.date)

    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path, index=False)
