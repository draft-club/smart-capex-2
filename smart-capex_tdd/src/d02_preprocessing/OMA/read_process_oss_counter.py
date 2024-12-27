"""Read process oss counter"""
import pandas as pd
from src.d01_utils.utils import add_logging_info


@add_logging_info
def preprocess_oss_weekly_from_capacity(df_oss_weekly_from_capacity, cell_filter, use_case):
    """
    Function who get weekly preprocess data from capacity pipeline and filter on use case
    TDD OR FDD

    Parameters
    ----------
    df_oss_weekly_from_capacity: pd.DataFrame
        Weekly Preprocess Data from Capacity
    cell_filter: list
        TDD
    use_case: str
        It can be 'FDD' or 'TDD'

    Returns
    -------
    df_oss_counter : pd.DataFrame
        Weekly Preprocess Data from Capacity filter on use case
    """
    data_to_concat = []
    for freq in cell_filter:
        if use_case == 'TDD':
            df_freq = df_oss_weekly_from_capacity[
                df_oss_weekly_from_capacity["cell_name"].str.contains(freq)]
        else:
            df_freq = df_oss_weekly_from_capacity[
                ~df_oss_weekly_from_capacity["cell_name"].str.contains(freq)]
        data_to_concat.append(df_freq)
    df_oss_counter = pd.concat(data_to_concat)
    return df_oss_counter
