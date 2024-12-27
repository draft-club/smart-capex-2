from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def preprocess_oss_counter(cell_technology: str,
                           raw_oss_counter_data_input: Input[Dataset],
                           processed_oss_counter_data_output: Output[Dataset]):
    """Preprocess OSS counter data and save it as a parquet file for the vertex pipeline.

    Args:
        cell_technology (str): It holds the cell technology type (e.g., "3G").
        raw_oss_counter_data_input (Input[Dataset]): It holds the input dataset containing raw OSS counter dataframe.
        processed_oss_counter_data_output (Output[Dataset]): It holds the output dataset to store
                                                                the processed OSS counter dataframe.

    """
    # Imports
    import pandas as pd

    # Load Data
    df_oss_counter = pd.read_parquet(raw_oss_counter_data_input.path)
    print("df_oss_counter shape", df_oss_counter.shape)

    # Function: preprocessing_oss_counter_3g_weeekly & preprocessing_oss_counter_4g_weeekly
    df_oss_counter["cell_tech"] = cell_technology
    df_oss_counter['cell_tech'] = df_oss_counter['cell_tech'].astype("category")

    df_oss_counter["date"] = pd.to_datetime(df_oss_counter["date"])

    # Function: create_dates_variables_from_week_period
    df_oss_counter['week_period'] = df_oss_counter['week_period'].astype(float).astype(int)
    df_oss_counter['week'] = df_oss_counter['week_period'].apply(lambda x: str(x)[4:]).astype(int)
    df_oss_counter['month'] = (df_oss_counter['date'].dt.month).astype(int)
    df_oss_counter['year'] = df_oss_counter['week_period'].apply(lambda x: str(x)[0:4]).astype(int)

    df_oss_counter.reset_index(drop=True, inplace=True)

    df_oss_counter.to_parquet(processed_oss_counter_data_output.path)
