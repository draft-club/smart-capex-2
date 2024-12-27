from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_traffic_weekly_kpis(project_id: str,
                             location: str,
                             table_id: str,
                             query_results_data_output: Output[Dataset]):
    """Load traffic weekly KPIs from BigQuery and saves the results as a parquet file for vertex pipeline.

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
            SELECT site_id, cell_tech, cell_band, week_date, week_period, total_data_traffic_dl_gb, total_voice_traffic_kerlands
            FROM {table_id}
        """

    df_query = client.query(query_statement).to_dataframe()
    df_query["week_date"] = pd.to_datetime(df_query["week_date"]).dt.date

    print("df_query_result shape: ", df_query.shape)

    df_query.to_parquet(query_results_data_output.path)
