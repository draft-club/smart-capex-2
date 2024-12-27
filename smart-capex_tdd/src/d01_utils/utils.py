"""Utils functions."""
import argparse
import configparser
import functools
import json
import logging.config
import math
import os
import pickle
import time
from contextlib import contextmanager
from datetime import datetime

import unicodedata
import numpy as np
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from src.d00_conf.conf import conf

load_dotenv()
R = 6371.0

# initiate logger
CONFIG_DIR = "./config"
LOG_DIR = "./logs"

# Read Running config.ini
config = configparser.ConfigParser()
config.read('running_config.ini')


def setup_logging():
    """
    The setup_logging function configures the logging system based on the environment variable ENV.
    It loads the appropriate logging configuration file, sets up a timestamped log file,
    and then removes old log files if there are more than five.

    Returns
    -------
    Configured logging system with a new log file and potentially removed old log files.
    """
    log_configs = {"dev": "logging.dev.ini", "prod": "logging.prod.ini"}
    config_logs = log_configs.get(os.environ["ENV"], "logging.dev.ini")
    config_path = "/".join([CONFIG_DIR, config_logs])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"},
    )
    remove_logs_files()


def remove_logs_files():
    """
    The remove_logs_files function deletes old log files in the LOG_DIR directory if there are more
    than five log files, keeping only the five most recent ones.

    Returns
    -------
    The function does not return any value but removes old log files from the LOG_DIR directory.
    """
    files = [os.path.join(LOG_DIR, file) for file in os.listdir(LOG_DIR)]
    if len(files) > 5:
        files.sort(key=os.path.getmtime, reverse=True)
        for file_to_delete in files[5:]:
            logging.info(f"Removing {file_to_delete}")
            os.remove(file_to_delete)


def savesample(df, name, nrow=1000):
    """
    The savesample function takes a DataFrame, samples a specified number of rows, and saves the
    sample to a CSV file with a specified name and semicolon as the delimiter.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to sample from
    name: str
        The name of the output CSV file.
    nrow: int
        The number of rows to sample (default is 1000).

    Returns
    -------
    A CSV file containing the sampled rows from the DataFrame.
    """
    df.sample(n=nrow).to_csv(name, sep=";")


# function to count execution time of a function
@contextmanager
def timer(title):
    """
    The timer function is a context manager that measures and prints the execution time of a
    code block.

    Parameters
    ----------
    title: str
        A string that describes the task being timed.
    Returns
    -------
    Prints the title and the elapsed time in seconds
    """
    # Gives execution time of a function
    t0 = time.time()
    yield
    print(f"{title}  - done ine {time.time() - t0:.0f}s")


def get_band_from_frequency(x):
    """
    The get_band_from_frequency function generates a string representing a cell band based on the
    input frequency value.

    Parameters
    ----------
    x: int
        A numerical value representing the frequency.

    Returns
    -------
    res: str
        A string in the format 'Lxx', where 'xx' are the first two digits of the integer part of the
        input frequency.

    """
    res = 'L' + str(int(x))[:2]
    return res


def one_hot_encoder(df, cols=None, nan_as_category=True):
    """
    The one_hot_encoder function converts categorical variables in a DataFrame into a series of
    binary (one-hot) encoded columns. It can handle missing values by treating them as a separate
    category.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data to be encoded.
    cols: list
         A list of column names to be one-hot encoded. If None, all categorical columns will be
         encoded.
    nan_as_category: bool
        A boolean indicating whether to treat NaN values as a separate category.

    Returns
    -------
    df: pd.DataFrame
        The DataFrame with one-hot encoded columns.
    new_columns: list
        A list of the new columns added to the DataFrame.
    """
    original_columns = list(df.columns)

    if cols is None:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    else:
        categorical_columns = cols
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# removes accent from a text variable
def remove_accent(text):
    """Remove tildes from a text variable"""
    try:
        text = str(text)
    except NameError:  # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text) \
        .encode('ascii', 'ignore') \
        .decode("utf-8")
    return str(text)


# Include a prefix to columns
def insert_prefix_to_columns(df, prefix, columns_exclude):
    """
    The insert_prefix_to_columns function adds a specified prefix to the names of columns in a
    DataFrame, excluding certain columns.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame whose columns need to be renamed
    prefix: str
        The prefix string to be added to the column names
    columns_exclude: list
        A list of column names that should not be prefixed.

    Returns
    -------
    df: pd.DataFrame
        The DataFrame with the specified prefix added to the column names, excluding the columns
        listed in columns_exclude.
    """
    df_dict = df.columns.values
    for name in df_dict:
        if name in columns_exclude:
            continue
        df.rename(columns={name: prefix + "_" + name}, inplace=True)
    return df


def get_week_period(year, week):
    """
    The get_week_period function generates a string representing a specific week of a given year in
    the format "YYYYWW". It ensures that the week number is always two digits by padding with a
    leading zero if necessary.

    Parameters
    ----------
    year: int
        an integer representing the year
    week: int
        an integer representing the week number.

    Returns
    -------
    A string in the format "YYYYWW" representing the specified week of the given year.
    """
    year = str(year)
    week = str(week)
    if len(week) == 1:
        week = '0' + week
    return year + week


def createdirectory(path):
    """
    The createdirectory function checks if a specified directory path exists, and if it does not,
    it creates the directory

    Parameters
    ----------
    path: str
        A string representing the directory path to be checked and potentially created.

    Returns
    -------
    The function does not return any value. It either creates the directory or does nothing if the
    directory already exists
    """
    if not os.path.exists(path):
        os.mkdir(path)


def getlastfile(path, pattern=""):
    """
    The getlastfile function retrieves the last file in a specified directory that matches a given
    pattern. It lists all files, filters them by the pattern, sorts them, and returns the last
    file in the sorted list

    Parameters
    ----------
    path: str
        The directory path where the files are located
    pattern: str
        A string pattern to filter the files (optional, default is an empty string)
    Returns
    -------
    The name of the last file in the sorted list that matches the given pattern
    """
    files = os.listdir(path)
    files = [f for f in files if pattern in f]
    files.sort()
    return files[-1]


def get_last_folder(path):
    """
    The get_last_folder function identifies and returns the most recently modified folder within a
    specified directory.

    Parameters
    ----------
    path: str
         A string representing the directory path where the function will search for folders.
    Returns
    -------
    latest_folder: str
        The function returns a string representing the path of the most recently modified folder
        within the specified directory.

    """
    folders = [os.path.join(path, folder) for folder in os.listdir(path)]
    latest_folder = max(folders, key=os.path.getmtime)
    print(latest_folder)
    return latest_folder


def round_50(x, base=50):
    """
    The round_50 function rounds a given number to the nearest multiple of 50.

    Parameters
    ----------
    x: int
        The number to be rounded
    base: int
        The base multiple to round to, default is 50
    Returns
    -------
    The function returns the number x rounded to the nearest multiple of base.
    """
    return base * round(x / base)


def get_band(x):
    """
    The get_band function extracts a specific part of a string based on delimiters.
    It handles cases where the input string is either a single character or contains multiple
    segments separated by hyphens and underscores.

    Parameters
    ----------
    x: str
        A string that may contain segments separated by hyphens and underscores.
    Returns
    -------
    x: str
        A string representing the extracted segment or the original input if no segments are found.

    Examples
    --------
    >>> get_band("L12-34_56")
    "34"
    >>> get_band("L12")
    "L12"
    >>> get_band("L12_34")
    "L12_34"

    """
    if len(x) == 1:
        return x
    try:
        return x.split('-')[-2].split('_')[-1]
    except IndexError:
        return x


def check_path(directory):
    """
    The check_path function ensures that a specified directory exists by creating it if it does
    not already exist.

    Parameters
    ----------
    directory: str
        A string representing the path of the directory to check or create.
    Returns
    -------
    The function does not return any value. It either ensures the directory exists or creates it if
    it does not

    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_csv(df, path, name, separator):
    """
    The write_csv function saves a DataFrame to a CSV file with a specified separator,
    ensuring the file has a .csv extension.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be saved.
    path: str
        The directory path where the CSV file will be saved.
    name: str
        The name of the CSV file
    separator: str
        The delimiter to use in the CSV file
    Returns
    -------
    A CSV file saved at the specified path with the given name and separator.
    """
    if '.csv' not in name:
        name = name + '.csv'
    df.to_csv(os.path.join(path, name), sep=separator, index=False)


def chdir_path(directory):
    """
    The chdir_path function changes the current working directory to the specified directory.
    Parameters
    ----------
    directory: str
        he path to the directory to which the current working directory should be changed
    Returns
    -------
    The function does not return any value.It changes the current working directory as a side effect
    """
    os.chdir(directory)


def write_model(model, path, name, conf_exec_time):
    """
    The write_model function saves a machine learning model to a specified directory with a given
    name. It ensures the file has a .sav extension and changes the working directory to a specified
    path before saving the model using the pickle module.

    Parameters
    ----------
    model: model
        The machine learning model to be saved.
    path: str
        The directory path where the model should be saved.
    name: str
        The name of the file to save the model as.
    conf_exec_time: str
         A subdirectory name, typically representing the execution time.

    Returns
    -------
    The function does not return any value. It saves the model to a file as a side effect.
    """
    if '.sav' not in name:
        name = name + '.sav'
    chdir_path(os.path.join(path, conf_exec_time))
    with open(name, 'wb') as file:
        pickle.dump(model, file)


def read_model(path, name, conf_path_model):
    if '.sav' not in name:
        name = name + '.sav'
    list_subfolders_with_paths = [f.name for f in os.scandir(os.path.join(
        conf_path_model, 'activation_model'))
                                  if f.is_dir()]
    last_model = max(list_subfolders_with_paths)
    print("The last model is from : " + last_model)
    chdir_path(os.path.join(path, last_model))
    with open(name, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


def correct_bands_upgraded(row):
    """
    The correct_bands_upgraded function takes a string input representing a band and returns a
    corrected band value based on predefined mappings.

    Parameters
    ----------
    row: str
        a string representing a band.
    Returns
    -------
    row: str
        The corrected band value as a string, or the original input if no conditions are met
    """
    if row == 'G9-L8':
        return 'L8'
    if row == 'G9-L26':
        return 'L26'
    if row == 'L26-U9':
        return 'L26'
    return row


def correct_tech_upgraded(row):
    """
    The correct_tech_upgraded function takes a string representing a technology upgrade path and
    returns the final technology in the path. It corrects specific upgrade paths to ensure the
    final technology is accurately represented.

    Parameters
    ----------
    row: str
        a string representing a technology upgrade path (e.g., '2G-4G', '4G-3G', '3G').

    Returns
    -------
    row: str
        The function returns a string representing the final technology in the upgrade path
    """
    if row == '2G-4G':
        return '4G'
    if row == '4G-3G':
        return '4G'
    return row


def get_month_year_period(d):
    """
    The get_month_year_period function takes a date object and returns a string in the format
    "YYYYMM". If the input is null, it returns the input as is

    Parameters
    ----------
    d: datetime

    Returns
    -------
    A string in the format "YYYYMM" or the input if it is null.
    """
    if pd.isnull(d):
        return d
    month = f'{d.month:02d}'
    year = f'{d.year:04d}'
    return year + month


def truncate_table(engine, table_name):
    """
    The truncate_table function deletes all rows from a specified table in a SQL database
    using SQLAlchemy.

    Parameters
    ----------
    engine: str
        SQLAlchemy engine object connected to the database.
    table_name: str
        Name of the table to be truncated.
    Returns
    -------
    The function does not return any value. It performs an action on the database.
    """
    session_ = sessionmaker(bind=engine)
    session = session_()
    # session.execute("SET GLOBAL FOREIGN_KEY_CHECKS = 0;")
    session.execute("TRUNCATE TABLE " + table_name)
    # session.execute("SELECT 'Hello World!' AS hello")
    session.commit()
    session.close()


def write_sql_database(df, table_name, if_exists='truncate'):
    """
    The write_sql_database function writes a DataFrame to a specified table in a MySQL database.
    It connects to the database using credentials from environment variables and can either
    truncate the table before writing or append to it based on the if_exists parameter

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be written to the database.
    table_name: str
        The name of the table in the database
    if_exists: str
        Action to take if the table already exists ('truncate' or 'append'). Default is 'truncate'

    Returns
    -------
    The function does not return any value. It performs an action on the database by writing the
    DataFrame to the specified table.

    """
    mysqluser = os.environ.get('MYSQLUSER')
    mysqlpassword = os.environ.get('MYSQLPASSWORD')
    mysqlport = os.environ.get('MYSQLPORT')
    mysqldatabase = os.environ.get('MYSQLDATABASE')
    mysqlhost = os.environ.get('MYSQLHOST')
    conn_uri = \
        f"mysql+pymysql://{mysqluser}:{mysqlpassword}@{mysqlhost}:{mysqlport}/{mysqldatabase}"
    engine = sqlalchemy.create_engine(conn_uri)
    if if_exists == "truncate":
        truncate_table(engine, table_name)
        if_exists = 'append'
    df.to_sql(table_name,
              con=engine,
              if_exists=if_exists,
              index=False)


def write_csv_sql(df, name, separator='|', path="", **kwargs):
    """
    The write_csv_sql function saves a DataFrame to both a CSV file and a SQL database table.
    It allows for optional parameters to control whether the DataFrame is saved as a CSV,
    written to a SQL table, or both

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be saved.
    name: str
        The base name for the CSV file and SQL table
    separator: str
        The delimiter to use in the CSV file (default is '|').
    path: str
        The directory path where the CSV file will be saved (default is "").

    Returns
    -------
    he function does not return any value. It performs actions to save the DataFrame to a CSV file
    and/or a SQL database table.

    """
    if kwargs.get('sql', True):
        if '.csv' in name:
            name = name.split(".csv")[0]
        write_sql_database(df, name, kwargs.get('if_exists', 'replace'))
    if kwargs.get('csv', True):
        write_csv(df=df, path=path, name=name, separator=separator)


def payback_of_investment(investment, cashflows):
    """
    The payback_of_investment function calculates the payback period, which is the time required for
    an investment to recover its initial cost from the cumulative cashflows.

    Parameters
    ----------
    investment: float
        The initial cost of the investment (a positive float).
    cashflows: float
        A list of cashflows (positive floats) expected over time.

    Returns
    -------
    The payback period as a float, representing the time required to recover the initial investment.

    Examples
    --------
    >>> payback_of_investment(200.0, [60.0, 60.0, 70.0, 90.0])
    3.1111111111111112
    """
    total, years, cumulative = 0.0, 0, []
    if not cashflows or (sum(cashflows) < investment):
        raise ValueError("insufficient cashflows")
    for cashflow in cashflows:
        total += cashflow
        if total < investment:
            years += 1
        cumulative.append(total)
    a = years
    b = investment - cumulative[years - 1]
    c = cumulative[years] - cumulative[years - 1]
    return a + (b / c)


def payback(cashflows):
    """The payback period refers to the length of time required
       for an investment to have its initial cost recovered.

       (This version accepts a list of cashflows)

       payback([-200.0, 60.0, 60.0, 70.0, 90.0])
       3.1111111111111112
    """
    investment, cashflows = cashflows[0], cashflows[1:]
    if investment < 0:
        investment = -investment
    return payback_of_investment(investment, cashflows)


# Write dictionary object on disk
def save_json(dictobj, name, space=4):
    """
    The save_json function writes a dictionary object to a JSON file with specified indentation.
    It handles file writing errors gracefully by returning an error message

    Parameters
    ----------
    dictobj: dict
        The dictionary object to be saved as a JSON file.
    name: str
        The name of the file where the JSON data will be saved.
    space: int
        The number of spaces for indentation in the JSON file (default is 4).
    Returns
    -------
    Returns None if the file is saved successfully.
    Returns an error message string if an OSError occurs.
    """
    try:
        with open(name, 'w', encoding='utf-8') as outfile:
            json.dump(dictobj, outfile, indent=space)
    except OSError:
        return "Une erreur est survenue! Enregistrement du fichier de configuration"
    return None


# Read json file from disk
def load_json(name):
    """
    The load_json function reads a JSON file and returns its content as a Python dictionary.
    If an IOError occurs, it prints the error and returns a specific error message in French.

    Parameters
    ----------
    name: str
        The name of the JSON file to be read

    Returns
    -------
    Returns a dictionary containing the JSON file content if successful.
    Returns an error message "Une erreur est survenue!:
    Lecture du fichier de configuration" if an IOError occurs.

    """
    try:
        with open(name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except IOError as e:
        print(e)
        return "Une erreur est survenue!: Lecture du fichier de configuration"


def get_running_config(configfile=None):
    """
    The get_running_config function reads a configuration file and returns a configuration object.

    Parameters
    ----------
    configfile : str
        The path to the configuration string representing the path to the configuration file.

    Returns
    -------
    config: ConfigParser object
        A ConfigParser object containing the parsed configuration data

    """
    config_object = configparser.ConfigParser()
    config_object.read(configfile)
    return config_object


def read_file(source_directory, data_directory, filename, sheet_name):
    """
    Read file and convert it to dataframe, geodataframe or obj (depending on the extension)

    Parameters
    ----------
    source_directory : str
        source directory of the whole project
    data_directory : str
        directory that contains datasets
    filename : str
        dataset name
    sheet_name: str
        name of the sheet in excel
    Returns
    -------
    df: pd.DataFrame
        object (pickle)
    """
    filepath = os.path.join(source_directory, data_directory, filename)
    print(filepath)
    obj = None
    if filename.endswith("xlsx"):
        obj = pd.read_excel(filepath, sheet_name=sheet_name,
                            header=0, engine="openpyxl",
                            index_col=None)
    elif filename.endswith("csv"):
        obj = pd.read_csv(filepath, sep=conf["CSV"]["SEP"], index_col=None)
        if obj.shape[1] == 1:
            obj = pd.read_csv(filepath, sep=",", index_col=None)  #
    elif filename.endswith("zip"):
        obj = pd.read_csv(filepath, index_col=None, compression='zip')
    elif filename.endswith("pkl"):
        with open(filepath, "rb") as file:
            obj = pickle.load(file)
    return obj


def compute_distance(lat1, lon1, lat2, lon2):
    """
    The compute_distance function calculates the great-circle distance between two points on the
    Earth's surface given their latitude and longitude in degrees.

    Parameters
    ----------
    lat1: float
        Latitude of the first point in degrees.
    lon1: float
        Longitude of the first point in degrees
    lat2: float
         Latitude of the second point in degrees
    lon2: float
        Longitude of the second point in degrees

    Returns
    -------
    distance: float
        The function returns the distance between the two points in kilometers.

    Examples
    --------
    >>> compute_distance(52.2296756, 21.0122287, 41.8919300, 12.5113300)
    1318.1388157320334

    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def npv_since_2nd_years(rate, values):
    """
    The npv_since_2nd_years function calculates the Net Present Value (NPV) of a series of
    cash flows, starting from the second year, by discounting each cash flow at a given rate.

    Parameters
    ----------
    rate: float
        The discount rate used to calculate the present value of future cash flows.
    values: list
        A list or array of cash flow values.

    Returns
    -------
    The function returns the NPV as a single numerical value
    """
    values = np.asarray(values)
    values = np.nan_to_num(values)
    return (values / (1 + rate) ** np.arange(1, len(values) + 1)).sum(axis=0)


def generate_bp_years(start_year):
    """
    Returns a list of date +5 of start_year

    Parameters
    ----------
    start_year: int
        Examples: 2023

    Returns
    -------
    list: list of integer
        Examples: [2024,2025,2026,2027,2028]
    """
    return (start_year + np.arange(0, conf["NPV"]["TIME_TO_COMPUTE_NPV"]) * 1).tolist()


def parse_arguments():
    """
    The parse_arguments function sets up an argument parser to handle command-line arguments,
    specifically for a file path to country parameters, and logs the parsed arguments

    Returns
    -------
    args: argparse.Namespace
        Returns an argparse.Namespace object containing the parsed arguments.

    Examples
    --------
    >>> args = parse_arguments()
    # Command-line usage:
    # python script.py --path_to_country_parameters='path/to/file.json'

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_country_parameters', type=str, required=False,
                        default='oma.json')
    args = parser.parse_args()
    logging.info(args)
    return args


def calculate_execution_time(func):
    """
    The calculate_execution_time function is a decorator that measures and logs the execution time
    of the decorated function.

    Parameters
    ----------
    func: function
        The function to be decorated and whose execution time is to be measured.

    Returns
    -------
    The result of the original function func.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {execution_time} seconds to execute")
        return result

    return wrapper


def add_logging_info(func):
    """
    The add_logging_info function is a decorator that logs the start and end of the execution of the
    decorated function, using the function's second line of the docstring as the description.

    Parameters
    ----------
    func: function
        The function to be decorated.
    Returns
    -------
    The result of the original function func.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        function_description = func.__doc__.split('\n')[1].strip()
        logging.info(f"Start: {function_description} ...")
        result = func(*args, **kwargs)
        logging.info(f"End: {function_description}")
        return result

    return wrapper


def super_decorator(func):
    """
    The super_decorator function is a decorator that logs the start and end of a function's
    execution, checks if the function is enabled or disabled based on a configuration file,
    and logs the execution time.

    Parameters
    ----------
    func: function
        The function to be decorated.

    Returns
    -------
    The result of the decorated function if it is enabled.
    None if the function is disabled or not found in the configuration.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_description = func.__doc__.split('\n')[1].strip()
        logging.info(f"Start: {function_description} ...")
        logging.info(f"Function name is: {func.__name__}")
        result = None
        if func.__name__ in config['Pipeline']:
            if int(config['Pipeline'][func.__name__]) == 2:
                logging.info(f"Function {func.__name__} is enabled")
                result = func(*args, **kwargs)
            elif int(config['Pipeline'][func.__name__]) == 0:
                logging.info(f"Function {func.__name__} is disabled")
                result = None
        else:
            result = None

        logging.info(f"End: {function_description}")
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {execution_time} seconds to execute")
        return result

    return wrapper
