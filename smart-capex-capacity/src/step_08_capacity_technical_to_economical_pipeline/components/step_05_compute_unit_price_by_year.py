from kfp.dsl import Dataset, Input, Output, component

from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def compute_unit_price_by_year(time_to_compute_npv: int,
                               last_year_of_oss: int,
                               unit_prices_data_input: Input[Dataset],
                               predicted_increase_in_traffic_by_the_upgrade_data_input: Input[Dataset],
                               predicted_increase_in_traffic_by_the_upgrade_data_output: Output[Dataset]):
    """
    Compute unit prices by year, add them to the predicted increase in traffic dataset 
    and save the resulted DataFrame for Vertex pipeline.

    Args:
        time_to_compute_npv (int): It holds number of years to compute NPV.
        last_year_of_oss (int): It holds the last year of OSS.
        unit_prices_data_input (Input[Dataset]): It holds input dataset containing unit prices DataFrame.
        predicted_increase_in_traffic_by_the_upgrade_data_input (Input[Dataset]): It holds input dataset containing
                                                                    predicted increase in traffic by the upgrade DataFrame.
        predicted_increase_in_traffic_by_the_upgrade_data_output (Output[Dataset]): It holds output dataset to store 
                                                                        the updated predicted increase in traffic DataFrame.
    """
    import numpy as np
    import pandas as pd

    df_predicted_increase_in_traffic_by_the_upgrade = pd.read_parquet(
                                                            predicted_increase_in_traffic_by_the_upgrade_data_input.path)
    df_unit_prices = pd.read_parquet(unit_prices_data_input.path)

    df_predicted_increase_in_traffic_by_the_upgrade["unit_price_data_mobile"] = 0
    df_predicted_increase_in_traffic_by_the_upgrade["unit_price_voice_min"] = 0

    start_year = last_year_of_oss + 1
    years = (start_year + np.arange(0, time_to_compute_npv) * 1).astype(str).tolist()

    df_unit_prices = df_unit_prices.set_index("year")

    for year in years:
        unit_price_data_mobile = df_unit_prices.loc[year, "ppm_data"]

        unit_price_voice_min = df_unit_prices.loc[year, "ppm_voice"]

        df_predicted_increase_in_traffic_by_the_upgrade.loc[
                                                    df_predicted_increase_in_traffic_by_the_upgrade['year'] == str(year),
                                                    'unit_price_data_mobile'] = unit_price_data_mobile

        df_predicted_increase_in_traffic_by_the_upgrade.loc[
                                                    df_predicted_increase_in_traffic_by_the_upgrade['year'] == str(year),
                                                    'unit_price_voice_min'] = unit_price_voice_min

    df_predicted_increase_in_traffic_by_the_upgrade['unit_price_data_mobile_with_the_decrease'] = \
                                                df_predicted_increase_in_traffic_by_the_upgrade["unit_price_data_mobile"]
    df_predicted_increase_in_traffic_by_the_upgrade['unit_price_voice_with_the_decrease'] = \
                                                    df_predicted_increase_in_traffic_by_the_upgrade["unit_price_voice_min"]

    df_predicted_increase_in_traffic_by_the_upgrade.to_parquet(predicted_increase_in_traffic_by_the_upgrade_data_output.path)
