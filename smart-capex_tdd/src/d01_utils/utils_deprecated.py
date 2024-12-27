import getpass
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from dateutil import rrule

from src.d00_conf.conf import conf


def run_if_config_enabled(section, config_file):
    """
    The run_if_config_enabled function is a decorator generator that checks a configuration file to
    determine if a specific section should run. If the section is enabled, it logs a message and
    executes the decorated function.

    Parameters
    ----------
    section: str
        The section in the configuration file to check.
    config_file: configuration file object
        The configuration file object to read from.

    Returns
    -------
    If the section is enabled, the original function's output is returned.
    If the section is not enabled, None is returned.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if config_file.getboolean(section, 'run'):
                logging.info("%s is enabled in running_config.ini", section)
                return func(*args, **kwargs)
            return None

        return wrapper

    return decorator


def convert_yearweek_to_date_friday(year_week):
    """
    The convert_yearweek_to_date_friday function takes a year and week number in the format YYYYWW
    and returns the date of the Friday of that week.

    Parameters
    ----------
    year_week: int
        An integer representing the year and week number in the format YYYYWW.

    Returns
    -------
    date_formated: str
        A string representing the date of the Friday of the given week in the format YYYY-MM-DD.
    """
    year = int(str(year_week)[:4])
    week = int(str(year_week)[4:])
    first_day_of_the_week = datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    friday_date = first_day_of_the_week + timedelta(days=4)
    date_formated = friday_date.strftime("%Y-%m-%d")
    return date_formated


def prepare_neighbors_for_tdd():
    """
    The prepare_neighbors_for_tdd function reads a CSV file containing site and neighbor information
    processes it to include only specific columns, and creates a new column cluster_key by
    concatenating the neighbor columns with a prefix. It returns the processed DataFrame.

    Returns
    -------
    df_site_id_neighbors: pd.DataFrame
        A pandas DataFrame containing site_id, neighbour_1, neighbour_2, and cluster_key columns.

    """
    df_site_id_neighbors = pd.read_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'distance_cluster.csv'), sep='|')
    df_site_id_neighbors = df_site_id_neighbors[['site_id', 'neighbour_1', 'neighbour_2']]
    df_site_id_neighbors['cluster_key'] = 'sitedensification' + '_' + df_site_id_neighbors[
        'neighbour_1'] + '_' + df_site_id_neighbors['neighbour_2']

    return df_site_id_neighbors


def save_file(obj, filepath):
    """
    The save_file function saves a given DataFrame object to a specified file path in either CSV or
    GeoJSON format, handling potential file operation errors.

    Parameters
    ----------
    obj : pd.DataFrame or gpd.DataFrame, data to be saved
        A pandas DataFrame or a geopandas DataFrame containing the data to be saved
    filepath : str
        path of the saved file
    """
    if filepath.endswith("csv"):
        try:
            obj.to_csv(filepath, index=False, sep=conf["CSV"]["SEP"])
        except OSError:
            pass
    if filepath.endswith("geojson"):
        try:
            obj.to_file(filepath, driver='GeoJSON')
        except OSError:
            pass


def week_upgrade_processing(df_):
    """
    The week_upgrade_processing function processes a DataFrame to calculate the lag in weeks between
    two specified week periods for each row.

    Parameters
    ----------
    df_: pd.DataFrame
        A pandas DataFrame containing columns week_of_the_upgrade and week_period.
    Returns
    -------
    df_: pd.DataFrame
        A pandas DataFrame with an additional column lag_between_upgrade representing the difference
         in weeks between the two week periods.

    """
    df_['week_of_the_upgrade'] = df_['week_of_the_upgrade'].apply(int).apply(str)
    df_['week_period'] = df_['week_period'].apply(int).apply(str)
    df_['lag_between_upgrade'] = df_[['week_of_the_upgrade', 'week_period']].apply(
        lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)
    return df_


def read_from_sql_database(table_name, chunksize=None):
    """
    The read_from_sql_database function reads data from a specified SQL table into a pandas
    DataFrame, optionally in chunks, using user-specific MySQL connection settings.

    Parameters
    ----------
    table_name: str
        The name of the SQL table to read from.
    chunksize: int
        Optional; the number of rows to include in each chunk.
    Returns
    -------
    df: pd.DataFrame
        Returns a pandas DataFrame containing the data from the specified SQL table, or an iterator
         of DataFrame chunks if chunksize is specified

    """
    user = getpass.getuser()
    path = os.path.join('home', user, 'my.cnf')
    engine = sqlalchemy.create_engine('mysql+pymysql://',
                                      connect_args={'read_default_file': "/" + path})
    df = pd.read_sql(table_name,
                     con=engine,
                     chunksize=chunksize)
    return df


def remove_columns_started_with_unnamed(df):
    """
    The remove_columns_started_with_unnamed function removes all columns from a DataFrame whose
    names start with 'Unnamed'.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame from which columns starting with 'Unnamed' will be removed.

    Returns
    -------
    df: pd.DataFrame:
        DataFrame with columns starting with 'Unnamed' removed.

    """
    df = df.drop(
        df.columns[
            df.columns.str.startswith('Unnamed')], axis='columns')
    return df


def get_lag_between_two_week_periods(week_period_1, week_period_2):
    """
    The get_lag_between_two_week_periods function calculates the difference in weeks between two
    given week periods, considering the year and week number

    Parameters
    ----------
    week_period_1: str
        A string representing the first week period in 'YYYYWW' format
    week_period_2: str
        A string representing the second week period in 'YYYYWW' format

    Returns
    -------
    An integer representing the difference in weeks between the two week periods

    """
    week_period_1, week_period_2 = str(week_period_1), str(week_period_2)
    year1 = int(week_period_1[:4])
    week1 = int(week_period_1[-2:])
    year2 = int(week_period_2[:4])
    week2 = int(week_period_2[-2:])
    return - (53 * year1 + week1 - (53 * year2 + week2))


def convert_float_string(aux):
    """
    The convert_float_string function converts a numeric input to a string. If the input is null,
    it returns the input as is. If the input is a float, it converts it to an integer before
    converting to a string.

    Parameters
    ----------
    aux: float
        The input value which can be a float, integer, string, or None.
    Returns
    -------

    """
    if pd.isnull(aux):
        return aux
    try:
        aux1 = str(int(aux))
    except TypeError:
        aux1 = str(aux)
    return aux1


def weeks_between(start_date, end_date):
    """
    The weeks_between function calculates the number of weeks between two given dates using the
    rrule module from the dateutil library.

    Parameters
    ----------
    start_date: datetime
        The starting date as a datetime object.
    end_date: datetime
        The ending date as a datetime object.

    Returns
    -------
    The function returns an integer representing the number of weeks between the two dates

    """
    weeks = rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date)
    return weeks.count()


def plot_prb(df_predicted, df_traffic_forecasting, df_affected_cells, site_id, cell_band):
    """
    The plot_prb function visualizes the congestion of a specific cell over time, comparing actual
    and predicted values. It checks the presence of the cell in multiple dataframes,
    retrieves relevant data, and plots the congestion percentage against the week period.

    Parameters
    ----------
    df_predicted: pd.DataFrame
        DataFrame containing predicted congestion values.
    df_traffic_forecasting: pd.DataFrame
        DataFrame containing traffic forecasting data.
    df_affected_cells: pd.DataFrame
        DataFrame containing affected cell data.
    site_id: str
        String representing the site identifier
    cell_band: str
        String representing the cell band identifier

    Returns
    -------
    A plot showing the congestion percentage over time for the specified cell,
    including both actual and predicted values

    """
    # Check if cell is in df_affected_cells
    if not (df_affected_cells[['site_id', 'cell_band']].values == [site_id, cell_band]).all(
            axis=1).any():
        print("Not in df_affected_cells")

    # Check if cell is in df_traffic_forecasting
    if not (df_traffic_forecasting[['site_id', 'cell_band']].values == [site_id, cell_band]).all(
            axis=1).any():
        print("Not in df_traffic_forecasting")

    # Check if cell is in df_predicted
    if not (df_predicted[['site_id', 'cell_band']].values == [site_id, cell_band]).all(
            axis=1).any():
        print("Not in df_predicted")

    # Get unique week of upgrade
    week_upgrade = df_affected_cells[
        (df_affected_cells['site_id'] == site_id) & (df_affected_cells['cell_band'] == cell_band)][
        'week_of_the_upgrade'].unique().astype(int)
    if week_upgrade.shape[0] != 1:
        print(f'Not working cell : {site_id}')
        print(f"there is 2 upgrade week : {week_upgrade}")

    # Get index and data for plotting
    date_index = df_affected_cells[
        (df_affected_cells['site_id'] == site_id) & (df_affected_cells['cell_band'] == cell_band)][
        'week_period'].index(week_upgrade)
    list_date = df_affected_cells[
        (df_affected_cells['site_id'] == site_id) & (df_affected_cells['cell_band'] == cell_band)][
        'week_period'].tolist()
    list_prb = df_affected_cells[
        (df_affected_cells['site_id'] == site_id) & (df_affected_cells['cell_band'] == cell_band)][
        'cell_occupation_dl_percentage'].tolist()
    y_pred = \
        df_predicted[
            (df_predicted['site_id'] == site_id) & (df_predicted['cell_band'] == cell_band)][
            'y_test'].tolist()

    if len(list_date) != len(list_prb):
        print(f'Not working cell : {site_id}')
        print(f'Len list_date : {len(list_date)}')
        print(f'Len list_prb  : {len(list_prb)}')
        print(f'Len date_index+8 : {date_index + 8}')

    list_date_str = [str(x) for x in list_date]

    # Plotting
    _, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.plot(list_date_str, list_prb, ls='solid')
    plt.ylim([0, 100])
    plt.locator_params(axis='x', nbins=10)
    plt.plot(list_date_str, np.repeat(y_pred, len(list_date)))  # y_pred line
    plt.plot(np.repeat(week_upgrade.astype(str), 2), [0, 100])
    plt.xticks(rotation=45)
    plt.title(site_id + " " + cell_band)
    plt.xlabel('week_period')
    plt.ylabel('congestion')
    plt.show()
