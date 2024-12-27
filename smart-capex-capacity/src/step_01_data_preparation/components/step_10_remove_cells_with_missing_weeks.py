from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def remove_cells_with_missing_weeks(missing_weeks_threshold: int,
                                    oss_counter_data_without_unmounted_cells_data_input: Input[Dataset],
                                    oss_counter_data_without_missing_weeks_data_output: Output[Dataset],
                                    cells_not_to_consider_data_output: Output[Dataset]):
    """Remove cells with missing weeks beyond a `missing_weeks_threshold` and save the results 
        as parquet files for the vertex pipeline.

    Args:
        missing_weeks_threshold (int): It holds the threshold for the number of missing weeks.
        oss_counter_data_without_unmounted_cells_data_input (Input[Dataset]): It holds the input dataset containing OSS
                                                                                counter dataframe without unmounted cells.
        oss_counter_data_without_missing_weeks_data_output (Output[Dataset]): It holds the output dataset to store the OSS 
                                                                                counter dataframe without missing weeks.
        cells_not_to_consider_data_output (Output[Dataset]): It holds the output dataset to store the 
                                                                cells not to consider dataframe.
    """

    # Imports
    import pandas as pd

    # Load Data
    df_oss_counter = pd.read_parquet(oss_counter_data_without_unmounted_cells_data_input.path)
    print(f"df_oss_counter shape before: {df_oss_counter.shape}")

    def detect_missing_weeks(df_oss_one_cell, missing_weeks_threshold):
        """Detect cells with missing weeks beyond `missing_weeks_threshold` after trimming the first 10 week lines
            in case they include more missing weeks above the threshold.

        Args:
            df_oss_one_cell (pd.DataFrame): It holds the DataFrame for a single cell.
            missing_weeks_threshold (int): It holds the threshold for the number of missing weeks.

        Returns:
            pd.DataFrame: The DataFrame with an additional column indicating if the cell is excluded.
        """

        n_week_lines_at_the_beginning = 10
        # trim at the beginning
        df_oss_one_cell = df_oss_one_cell.reset_index(drop=True)
        df_series = df_oss_one_cell["date"].sort_values(ascending=True)
        df_series = df_series.iloc[:n_week_lines_at_the_beginning]
        df_series = df_series.diff()
        trimming_boolean = (df_series.dt.days > missing_weeks_threshold * 7).any()
        if trimming_boolean:
            # trim everything before triming_index including the index itself
            df_oss_one_cell.sort_values(by="date", ascending=True, inplace=True)
            df_oss_one_cell = df_oss_one_cell.iloc[n_week_lines_at_the_beginning:]

        df_oss_one_cell.reset_index(inplace=True, drop=True)
        df_oss_one_cell.sort_values(by="date", ascending=True, inplace=True)
        df_oss_one_cell["diff_in_days"] = df_oss_one_cell["date"].diff()
        df_oss_one_cell["cell_excluded"] = (df_oss_one_cell["diff_in_days"].dt.days > missing_weeks_threshold * 7).any()

        return df_oss_one_cell


    df_oss_counter = df_oss_counter.groupby("cell_name").apply(detect_missing_weeks,
                                                               missing_weeks_threshold=missing_weeks_threshold)
    df_oss_counter = df_oss_counter.reset_index(drop=True)

    print(f"df_oss_counter shape after: {df_oss_counter.shape}")

    df_cells_not_to_consider = df_oss_counter.loc[df_oss_counter["cell_excluded"],
                                                  ["cell_name"]].drop_duplicates(ignore_index=True)
    df_oss_counter = df_oss_counter[~df_oss_counter["cell_excluded"]]

    df_oss_counter.reset_index(drop=True, inplace=True)
    df_oss_counter.drop(columns=["diff_in_days", "cell_excluded"], inplace=True)

    df_oss_counter.to_parquet(oss_counter_data_without_missing_weeks_data_output.path)
    df_cells_not_to_consider.to_parquet(cells_not_to_consider_data_output.path)
