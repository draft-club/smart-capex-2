from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_capex_opex(project_id: str,
                         location: str,
                         capex_table_id: str,
                         opex_table_id: str,
                         capex_data_output: Output[Dataset],
                         opex_data_output: Output[Dataset]):
    """Load capex and opex data from BigQuery and save it as a parquet file for the vertex pipeline.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        capex_table_id (str): It holds the table ID for capex table.
        opex_table_id (str): It holds the table ID for opex table.
        capex_data_output (Output[Dataset]): It holds the output dataset to store the raw capex data.
        opex_data_output (Output[Dataset]): It holds the output dataset to store the raw opex data.

    """

    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    capex_query_statement = f"""
                            SELECT *
                                FROM {capex_table_id}
                                WHERE provider = "Huawei"
                            """
    opex_query_statement = f"""
                        SELECT *
                            FROM {opex_table_id}
                        """

    df_capex_query_result = client.query(capex_query_statement).to_dataframe()
    df_opex_query_result = client.query(opex_query_statement).to_dataframe()

    print("df_capex_query_result shape: ", df_capex_query_result.shape)
    print("df_opex_query_result shape: ", df_opex_query_result.shape)

    df_capex_query_result.to_parquet(capex_data_output.path)
    df_opex_query_result.to_parquet(opex_data_output.path)
