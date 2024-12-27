from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def handle_no_congested_cells(unique_congested_cells_data_input: Input[Dataset],
                              unique_site_features_data_input: Input[Dataset],
                              detected_cell_congestion_data_input: Input[Dataset],
                              no_congestion_data_output: Output[Dataset]):
    """It filters the no congested cells from the detected cell congestion data and merges with the site features data

    Args:
        unique_congested_cells_data_input (Input[Dataset]): It holds the unique congested cell_name
        unique_site_features_data_input (Input[Dataset]): It holds the unique site features (cell_tech, cell_band) data
        detected_cell_congestion_data_input (Input[Dataset]): It holds dataframet with the congestion column
        no_congestion_data_output (Output[Dataset]): It holds the no congestion data

    Returns:
        no_congestion_data_output (Output[Dataset]): It holds the no congestion data
    """

    # imports
    import pandas as pd

    # Load Data
    df_sites = pd.read_parquet(unique_site_features_data_input.path)
    df_cells = pd.read_parquet(detected_cell_congestion_data_input.path)
    df_unique_congested_cells = pd.read_parquet(unique_congested_cells_data_input.path)
    list_of_congested_cells = df_unique_congested_cells["cell_name"]

    df_cells_no_congestion = df_cells[df_cells["congestion"] == "NO_CONGESTION"]
    df_cells_no_congestion.drop_duplicates(subset=["cell_name"], keep="first", inplace=True)

    df_cells_no_congestion = df_cells_no_congestion[~df_cells_no_congestion["cell_name"].isin(list_of_congested_cells)]

    df_no_congestion = df_cells_no_congestion.merge(df_sites, how="left", on='site_id')
    df_no_congestion["bands_upgraded"] = ""
    df_no_congestion["tech_upgraded"] = ""

    print("df_no_congestion after: ", df_no_congestion.shape)

    df_no_congestion.to_parquet(no_congestion_data_output.path)
