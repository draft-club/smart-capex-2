from kfp.dsl import Dataset, Input, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def preprocess_capex_opex(project_id: str,
                          location: str,
                          processed_capex_table_id: str,
                          processed_opex_table_id: str,
                          capex_data_input: Input[Dataset],
                          opex_data_input: Input[Dataset]):
    """It preprocesses the capex and opex data and saves them to BigQuery.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        processed_capex_table_id (str): It holds the table ID for the processed capex in BigQuery.
        processed_opex_table_id (str): It holds the table ID for the processed opex in BigQuery.
        capex_data_input (Input[Dataset]):  It holds the input dataset containing the capex data.
        opex_data_input (Input[Dataset]): It holds the input dataset containing the opex data.
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    df_capex = pd.read_parquet(capex_data_input.path)
    df_opex = pd.read_parquet(opex_data_input.path)

    df_capex["bands"] = df_capex["action"].str.replace("Add ", "L")
    df_capex.rename(columns={"capex": "capex_cost"}, inplace=True)
    df_opex["bands"] = df_opex["action"].str.replace("Add ", "L")

    print("df_capex shape: ", df_capex.shape)
    print("df_opex shape: ", df_opex.shape)

    # Save to bigquery
    pandas_gbq.to_gbq(df_capex, processed_capex_table_id, project_id=project_id, location=location, if_exists='replace')
    pandas_gbq.to_gbq(df_opex, processed_opex_table_id, project_id=project_id, location=location, if_exists='replace')
