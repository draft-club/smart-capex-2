from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def preprocess_cells_bands(project_id: str,
                           location: str,
                           preprocessed_sites_data_input: Input[Dataset],
                           processed_sites_table_id: str,
                           processed_sites_cells_bands_data_output: Output[Dataset]):
    """Preprocess cell bands data and save it to BigQuery for other vertex pipelines and as a parquet file
        for the current pipeline. The cell_band values are mapped before dropping all the rows with NaNs.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        preprocessed_sites_data_input (Input[Dataset]): It holds the input dataset containing preprocessed site data.
        processed_sites_table_id (str): It holds the table ID for the processed sites in BigQuery.
        processed_sites_cells_bands_data_output (Output[Dataset]): It holds the output dataset to store 
                                                                    the processed site cell bands dataframe.

    """

    # Imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    df_sites = pd.read_parquet(preprocessed_sites_data_input.path)
    print("df_sites before columns", df_sites.columns)
    print("df_sites before shape", df_sites.shape)

    band_mapping_dict = {"UMTS2100": "U2100", "UMTS900": "U900"}

    # used replace instead of map to keep the unlisted values
    df_sites["cell_band"] = df_sites["cell_band"].replace(band_mapping_dict)
    print("df_sites cell_band", df_sites["cell_band"].unique())

    df_sites = df_sites.dropna(how="any")

    # Save to bigquery
    pandas_gbq.to_gbq(df_sites, processed_sites_table_id, project_id=project_id, location=location, if_exists='replace')

    df_sites.to_parquet(processed_sites_cells_bands_data_output.path)
