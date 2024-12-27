from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def merge_oss_counters(processed_3g_data_input: Input[Dataset],
                       processed_4g_data_input: Input[Dataset],
                       oss_counters_data_output: Output[Dataset]):
    """Merge 3G and 4G OSS counter data and save it as a parquet file for the vertex pipeline.

    Args:
        processed_3g_data_input (Input[Dataset]): It holds the input dataset containing processed 3G dataframe.
        processed_4g_data_input (Input[Dataset]): It holds the input dataset containing processed 4G dataframe.
        oss_counters_data_output (Output[Dataset]): It holds the output dataset to store the merged OSS counter dataframe.
    """
    # Imports
    import pandas as pd

    # Load Data
    df_processed_3g = pd.read_parquet(processed_3g_data_input.path)
    df_processed_4g = pd.read_parquet(processed_4g_data_input.path)

    df_oss_counter = pd.concat([df_processed_3g, df_processed_4g])
    print("df_oss_counter shape before", df_oss_counter.shape)

    df_oss_counter.to_parquet(oss_counters_data_output.path)
