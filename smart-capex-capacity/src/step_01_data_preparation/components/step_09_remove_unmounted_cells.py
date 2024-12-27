from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def remove_unmounted_cells(unmounted_weeks_threshold: int,
                           oss_counter_data_without_recent_cells_data_input: Input[Dataset],
                           oss_counter_data_without_unmounted_cells_data_output: Output[Dataset],
                           cells_not_to_consider_data_output: Output[Dataset]):
    """Removes cells that have been unmounted for a specified number of weeks and saves the results as parquet files.

    Args:
        unmounted_weeks_threshold (int): It holds the threshold for the number of weeks a cell has been unmounted.
        oss_counter_data_without_recent_cells_data_input (Input[Dataset]): It holds the input dataset containing 
                                                                            OSS counter data without recent cells.
        oss_counter_data_without_unmounted_cells_data_output (Output[Dataset]): It holds the output dataset to store the 
                                                                                OSS counter data without unmounted cells.
        cells_not_to_consider_data_output (Output[Dataset]): It holds the output dataset to store the cells not to consider dataframe.
    """

    # Imports
    import pandas as pd

    # Load Data
    df_oss_counter = pd.read_parquet(oss_counter_data_without_recent_cells_data_input.path)
    print("df_oss_counter shape before", df_oss_counter.shape)

    df_oss_counter["week_period"] = df_oss_counter["week_period"].astype(int)

    max_week_period = df_oss_counter["week_period"].max()

    cells_not_to_consider_index = df_oss_counter.groupby("cell_name")["week_period"].max() \
        [df_oss_counter.groupby("cell_name")["week_period"].max() < max_week_period - unmounted_weeks_threshold].index
    df_cells_not_to_consider = df_oss_counter[df_oss_counter["cell_name"].isin(cells_not_to_consider_index)]
    df_cells_not_to_consider = df_cells_not_to_consider[["cell_name"]].drop_duplicates(ignore_index=True)

    df_oss_counter = df_oss_counter[~df_oss_counter["cell_name"].isin(cells_not_to_consider_index)]

    print("df_oss_counter shape after", df_oss_counter.shape)

    df_oss_counter.to_parquet(oss_counter_data_without_unmounted_cells_data_output.path)
    df_cells_not_to_consider.to_parquet(cells_not_to_consider_data_output.path)
