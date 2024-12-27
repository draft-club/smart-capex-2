from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def merge_oss_counter_with_sites(project_id: str,
                                 location: str,
                                 processed_oss_counter_table_id: str,
                                 oss_counter_data_without_high_variation_data_input: Input[Dataset],
                                 processed_sites_data_input: Input[Dataset],
                                 processed_oss_counter_data_output: Output[Dataset]):
    """Merge OSS counter data with site data and save the results to BigQuery for the other vertex pipelines 
        and as a parquet file for the current pipeline.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        processed_oss_counter_table_id (str): It holds the table ID for the processed OSS counter data in BigQuery.
        oss_counter_data_without_high_variation_data_input (Input[Dataset]): It holds the input dataset containing OSS
                                                                                counter data without high variation in KPI.
        processed_sites_data_input (Input[Dataset]): It holds the input dataset containing processed site data.
        processed_oss_counter_data_output (Output[Dataset]): It holds the output dataset to store the merged
                                                                OSS counter dataframe.
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    df_oss_counter = pd.read_parquet(oss_counter_data_without_high_variation_data_input.path)
    df_sites = pd.read_parquet(processed_sites_data_input.path)

    print("df_oss_counter shape: ", df_oss_counter.shape)
    print("df_sites shape: ", df_sites.shape)

    # replaced `ville` with `department`
    df_sites = df_sites[["site_id", "cell_name", "cell_tech", "cell_band", "region", "department"]]

    df_merged = pd.merge(left=df_oss_counter, right=df_sites, how="inner", on=["cell_name", "cell_tech"])

    max_date_threshold = df_oss_counter.groupby("cell_tech")["date"].max().min()
    df_merged = df_merged[df_merged["date"] <= max_date_threshold]

    print("df_merged shape: ", df_merged.shape)

    df_merged.to_parquet(processed_oss_counter_data_output.path)

    # rename the column before saving into database
    df_merged = df_merged.rename(columns={"date": "week_date"})
    df_merged["week_date"] = pd.to_datetime(df_merged["week_date"]).dt.date

    df_merged["week_date"] = pd.to_datetime(df_merged["week_date"]).dt.date

    # Added this line to force the week_date column to be saved as DATE type not TIMESTAMP
    table_schema = [{"name": "week_date", "type": "DATE"}]

    # Save to bigquery
    pandas_gbq.to_gbq(df_merged, processed_oss_counter_table_id,
                      project_id=project_id, location=location,
                      if_exists='replace', table_schema=table_schema)
