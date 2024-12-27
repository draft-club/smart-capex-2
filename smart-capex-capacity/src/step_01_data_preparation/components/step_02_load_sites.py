from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def load_sites(project_id: str,
               location: str,
               deployment_history_table_id: str,
               raw_sites_data_output: Output[Dataset]):
    """Load site data from BigQuery and save it as a parquet file for the vertex pipeline.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        deployment_history_table_id (str): It holds the table ID for deployment history.
        raw_sites_data_output (Output[Dataset]): It holds the output dataset to store the raw site data.

    """

    # imports
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)

    # temporary: removed site_id MO0034 (site_area=null)
    def get_query_results(table_id, interval=1):
        """Fetch query results from BigQuery based on the provided OSS counter table IDs and cell technology.

        Args:
            table_id (str): The table ID of the BigQuery table.

        Returns:
            pd.DataFrame: The DataFrame resulting from the query.
        """

        query_statement = f"""
            SELECT site_id, site_latitude, site_longitude, site_commune, site_department, cell_name, cell_tech, 
            cell_band, site_region, site_gestionnaire, site_area
            FROM {table_id}
            WHERE (week_start_date >= TIMESTAMP_SUB(
                (SELECT max(week_start_date) 
                FROM {table_id}), INTERVAL {interval} DAY))
                AND (cell_tech = '3G' or cell_tech = '4G')
                AND site_status = 'ENABLED'
                AND site_area is not null
        """  
        query_results = client.query(query_statement).to_dataframe()
        return query_results
    
    df_sites = get_query_results(deployment_history_table_id)

    sites_table_mapping = {"site_latitude": "latitude", "site_longitude": "longitude", "site_commune": "commune",
                           "site_department": "department", "site_region": "region"}
    df_sites = df_sites.rename(columns=sites_table_mapping)

    df_sites.loc[:, ["latitude", "longitude"]] = df_sites.loc[:, ["latitude", "longitude"]].astype(float)
    print(f"df_sites shape: {df_sites.shape}")

    df_sites.to_parquet(raw_sites_data_output.path)
