from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


 # pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def load_data(project_id: str,
              location: str,
              processed_oss_counter_table_id: str,
              traffic_weekly_kpis_data_output: Output[Dataset]):
    """It loads the data from GCP

    Args:
        project_id (str): It holds the project_id on GCP
        location (str): It holds the location assigned to the project on GCP
        processed_oss_counter_table_id (str): It holds the resource name of processed OSS Counter table on BigQuery
        traffic_weekly_kpis_data_output (Output[Dataset]): It holds the processed traffic weekly KPIs 
                                                            returned from BigQuery

    Returns:
        traffic_weekly_kpis_data_output: It holds the processed traffic KPIs returned from BigQuery
    """
    # imports
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    def get_query_results(table_id: str) -> pd.DataFrame:
        """ Get query results from BigQuery

        Args:
            table_id (str): table id 
        Returns:
            pd.DataFrame: query results
        """

        # Get Query from the last 2 years untill the max date (for offline data)
        query_statement = f"""SELECT week_date, cell_name, cell_tech, cell_band,
                                     site_id, year, week, week_period,
                                     total_voice_traffic_kerlands, total_data_traffic_dl_gb, average_number_of_users_in_queue, average_throughput_user_dl
                                FROM {table_id}
                            """

        df_query = client.query(query_statement).to_dataframe()

        return df_query

    df_traffic_weekly_kpis = get_query_results(table_id=processed_oss_counter_table_id)
    df_traffic_weekly_kpis["week_date"] = pd.to_datetime(df_traffic_weekly_kpis["week_date"]).dt.date
    df_traffic_weekly_kpis = df_traffic_weekly_kpis.rename(columns={"week_date": "date"})

    print("df_traffic_weekly_kpis shape: ", df_traffic_weekly_kpis.shape)

    df_traffic_weekly_kpis.to_parquet(traffic_weekly_kpis_data_output.path)
