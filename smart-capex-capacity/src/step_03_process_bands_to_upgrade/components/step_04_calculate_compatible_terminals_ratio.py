from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def calculate_compatible_terminals_ratio(b32_data_input: Input[Dataset],
                                         b32_aggregated_data_output: Output[Dataset]):
    """It calculates the compatible terminals ratio

    Args:
        b32_data_input (Input[Dataset]): It holds the b32 data
        b32_aggregated_data_output (Output[Dataset]): It holds the calculated compatible terminals ratio column

    Returns:
        b32_aggregated_data_output (Output[Dataset]):  It holds the calculated compatible terminals ratio column
    """    """"""
    # Imports
    import pandas as pd
    import numpy as np


    # Load Data
    df_b32 = pd.read_parquet(b32_data_input.path)

    max_file_date_for_each_site_indices = df_b32.groupby('site_id')['file_date'].idxmax()
    df_b32_aggregated = df_b32.loc[max_file_date_for_each_site_indices].reset_index(drop=True)

    # Cast with np.float64 to overcome the infinity caused by the division by Zero
    df_b32_aggregated["compatible_terminals_ratio"] = np.where(df_b32_aggregated["4g_devices"] == 0, 0, (
                                                      np.float64(df_b32_aggregated["band_32_devices"]) /
                                                      np.float64(df_b32_aggregated["4g_devices"])))

    df_b32_aggregated["file_date"] = pd.to_datetime(df_b32_aggregated["file_date"])
    df_b32_aggregated["4g_devices"] = df_b32_aggregated["4g_devices"].astype(float)
    df_b32_aggregated["band_32_devices"] = df_b32_aggregated["band_32_devices"].astype(float)

    print("df_b32_aggregated shape: ", df_b32_aggregated.shape)
    df_b32_aggregated.to_parquet(b32_aggregated_data_output.path)
