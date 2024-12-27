from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def detect_cell_congestion(predicted_traffic_kpis_data_input: Input[Dataset],
                           detected_cell_congestion_data_output: Output[Dataset]):
    """It detects the cell congestion based on the predicted traffic KPIs

    Args:
        predicted_traffic_kpis_data_input (Input[Dataset]): It holds the predicted traffic KPIs data
        detected_cell_congestion_data_output (Output[Dataset]): It holds dataframet with he congestion column

    Returns:
        detected_cell_congestion_data_output (Output[Dataset]): It holds dataframet with he congestion column
    """

    # imports
    import pandas as pd

    # Load Data
    df_predicted_traffic_kpis = pd.read_parquet(predicted_traffic_kpis_data_input.path)

    df_cells = df_predicted_traffic_kpis[['cell_name', 'week_date', 'cell_tech', 'cell_band', 'site_id', 'year',
                                          'week', 'week_period', 'total_voice_traffic_kerlands',
                                          'total_data_traffic_dl_gb', 'average_number_of_users_in_queue',
                                          'average_throughput_user_dl']]

    # Define the conditions for 3g and 4g congestion
    cell_technology_3g_check = df_cells["cell_tech"] == "3G"
    cell_technology_4g_check = df_cells["cell_tech"] == "4G"

    cell_technology_3g_congestion_check = ((df_cells["average_number_of_users_in_queue"] * 100 > 250) &
                                           (df_cells["average_throughput_user_dl"] / 1024 < 1))

    cell_technology_4g_congestion_check = ((df_cells["average_number_of_users_in_queue"] * 100 > 250) &
                                           (df_cells["average_throughput_user_dl"] / 1024 < 3))

    consecutive_zeros_exception_check = ((df_cells["average_number_of_users_in_queue"] != 9999) |
                                         (df_cells["average_throughput_user_dl"] != 9999))

    # Initialize the congestion status column
    df_cells["congestion"] = "NO_CONGESTION"

    # Assign congestion statuses based on the conditions
    df_cells.loc[cell_technology_3g_check & cell_technology_3g_congestion_check &
                consecutive_zeros_exception_check, "congestion"] = "3G_CONGESTION"
    df_cells.loc[cell_technology_4g_check & cell_technology_4g_congestion_check &
                 consecutive_zeros_exception_check, "congestion"] = "4G_CONGESTION"

    print("df_cells shape: ", df_cells.shape)
    print("df_cells value_counts: ", df_cells['congestion'].value_counts())

    df_cells.to_parquet(detected_cell_congestion_data_output.path)
