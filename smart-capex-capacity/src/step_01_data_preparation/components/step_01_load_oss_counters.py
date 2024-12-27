from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_oss_counters(project_id: str,
                      location: str,
                      oss_counters_table_id: str,
                      oss_counters_history_table_id: str,
                      cell_technology: str,
                      raw_oss_data_output: Output[Dataset]):
    """Load OSS counters data from BigQuery and save it as a parquet file for the vertex pipeline.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        oss_counters_table_id (str): It holds the table ID for current OSS counters.
        oss_counters_history_table_id (str): It holds the table ID for historical OSS counters.
        cell_technology (str): It holds the cell technology type (e.g., "3G").
        raw_oss_data_output (Output[Dataset]): The output dataset to store the raw OSS output DataFrame.
    """

    # imports
    import pandas as pd
    from google.cloud import bigquery

    def get_query_results(oss_counters_table_id, oss_counters_history_table_id, cell_technology):
        """Fetch query results from BigQuery based on the provided OSS counter table IDs and cell technology.

        Args:
            oss_counters_table_id (str): The table ID for current OSS counters.
            oss_counters_history_table_id (str): The table ID for historical OSS counters.
            cell_technology (str): The cell technology type (e.g., "3G").

        Returns:
            pd.DataFrame: The DataFrame resulting from the query.
        """

        query_statement = ""

        voice_traffic_kerlands_field = ""
        comma = ""

        if cell_technology == "3G":
            voice_traffic_kerlands_field = "total_voice_traffic_kerlands"
            comma = ","

        query_statement = f"""SELECT
                                  week As week_period,
                                  week_start_date As date,
                                  cell_name,
                                  {voice_traffic_kerlands_field}{comma}
                                  total_data_traffic_dl_gb,
                                  average_number_of_users_in_queue,
                                  average_throughtput_user_dl AS average_throughput_user_dl
                                FROM
                                  {oss_counters_table_id}

                                UNION ALL

                                SELECT
                                  week As week_period,
                                  week_start_date As date,
                                  cell_name,
                                  {voice_traffic_kerlands_field}{comma}
                                  total_data_traffic_dl_gb,
                                  users AS average_number_of_users_in_queue,
                                  throughput AS average_throughput_user_dl
                                FROM
                                  {oss_counters_history_table_id}"""

        query_results = client.query(query_statement).to_dataframe()
        return query_results

    client = bigquery.Client(project=project_id, location=location)
    df_oss_counters = get_query_results(oss_counters_table_id, oss_counters_history_table_id, cell_technology)
    df_oss_counters['date'] = pd.to_datetime(df_oss_counters['date']).dt.date

    print(f"df_oss_counters shape: {df_oss_counters.shape}")

    df_oss_counters.to_parquet(raw_oss_data_output.path)
