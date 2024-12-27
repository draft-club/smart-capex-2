from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def remove_cells_with_high_variation_in_kpi(variation_coefficient_threshold: float,
                                            oss_counter_data_without_missing_weeks_data_input: Input[Dataset],
                                            oss_counter_data_without_high_variation_in_kpi_data_output: Output[Dataset],
                                            cells_not_to_consider_data_output: Output[Dataset]):
    """Remove cells with variation coefficient greater than variation_coefficient_threshold 
        and save the results as parquet files for vertex pipeline.

    Args:
        df_oss_counter (pd.DataFrame): OSS counter dataframe
        variation_coefficient_threshold (float): The threshold for determining high variation.

    Returns:
        (pd.DataFrame): DataFrame with cells with high KPI variation removed.
        (list): List of cell names that were removed due to high KPI variation    
    """

    # Imports
    import numpy as np
    import pandas as pd

    # Load Data
    df_oss_counter = pd.read_parquet(oss_counter_data_without_missing_weeks_data_input.path)

    # 2G cell technology is removed
    cell_technology_kpi = {"3G": "total_voice_traffic_kerlands",
                           "4G": "total_data_traffic_dl_gb"}

    def detect_high_kpi_variation(df_oss_one_cell):
        """Detect cells with high KPI variation based on fixed `variation_coefficient_threshold`.

        Args:
            df_oss_one_cell (pd.DataFrame): It holds the DataFrame for a single cell.

        Returns:
            str or None: The cell name if it has high KPI variation, otherwise None.
        """

        cell_technology = df_oss_one_cell["cell_tech"].iloc[0]
        kpi = cell_technology_kpi[cell_technology]

        variation_coefficient = (np.std(df_oss_one_cell[kpi]) / (sum(df_oss_one_cell[kpi]) / len(df_oss_one_cell[kpi]))) * 100

        if variation_coefficient > variation_coefficient_threshold:
            return df_oss_one_cell["cell_name"].iloc[0]

        return None

    removed_cells = df_oss_counter.groupby("cell_name").apply(detect_high_kpi_variation).dropna().tolist()
    df_filtered_oss_counter = df_oss_counter[~df_oss_counter["cell_name"].isin(removed_cells)]

    # return df_filtered_oss_counter, removed_cells
    df_cells_not_to_consider = pd.DataFrame({"cell_name": removed_cells}).drop_duplicates(ignore_index=True)

    df_filtered_oss_counter.to_parquet(oss_counter_data_without_high_variation_in_kpi_data_output.path)
    df_cells_not_to_consider.to_parquet(cells_not_to_consider_data_output.path)
