from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def remove_recent_cells(recent_weeks_threshold: int,
                        merged_oss_counters_data_input: Input[Dataset],
                        oss_counter_data_without_recent_cells_data_output: Output[Dataset],
                        cells_not_to_consider_data_output: Output[Dataset]):
    """Remove cells with data from less than `recent_weeks_threshold` and save the results as parquet files 
        for vertex pipeline.

    Args:
        recent_weeks_threshold (int): It holds the threshold for the number of recent weeks.
        merged_oss_counters_data_input (Input[Dataset]): It holds the input dataset containing merged OSS counter data.
        oss_counter_data_without_recent_cells_data_output (Output[Dataset]): It holds the output dataset to store the 
                                                                                OSS counter dataframe without recent cells.
        cells_not_to_consider_data_output (Output[Dataset]): It holds the output dataset to store the dataframe that holds 
                                                                cells not to consider.
    """

    # Imports
    import pandas as pd

    # Load Data
    df_merged_oss_counters = pd.read_parquet(merged_oss_counters_data_input.path)

    # remove cells with less than 48 weeks
    df_merged_oss_counters["date"] = pd.to_datetime(df_merged_oss_counters["date"], format="%Y-%m-%d")
    df_dates = df_merged_oss_counters.groupby("cell_name").agg({"date": [min, max]})
    df_dates["diff"] = ((df_dates[('date', 'max')] - df_dates[('date', 'min')]).dt.days).astype(float)

    cells_not_to_consider_index = df_dates[df_dates["diff"] < recent_weeks_threshold * 7].index
    df_cells_not_to_consider = df_merged_oss_counters[df_merged_oss_counters["cell_name"].isin(cells_not_to_consider_index)]
    df_cells_not_to_consider = df_cells_not_to_consider[["cell_name"]].drop_duplicates(ignore_index=True)

    df_merged_oss_counters = df_merged_oss_counters[~df_merged_oss_counters["cell_name"].isin(cells_not_to_consider_index)]

    print("df_oss_counter shape after", df_merged_oss_counters.shape)

    df_merged_oss_counters.to_parquet(oss_counter_data_without_recent_cells_data_output.path)
    df_cells_not_to_consider.to_parquet(cells_not_to_consider_data_output.path)
