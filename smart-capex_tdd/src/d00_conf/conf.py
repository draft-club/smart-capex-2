""" Configuration file and load conf from country"""
import os
import datetime
import json


# Create an empty dictionary containing the global configuration
# Loaded in main from ****.json file

#global conf
conf = {}


def conf_loader(country_configuration):
    """
    Load configuration from country specific json file and intialize paths

    Parameters
    ----------
    country: str
        'OMA' or 'OCI' for example
    """
    file_path = os.path.join(os.path.dirname(__file__), country_configuration)
    with open(file_path, encoding='utf-8') as jsonfile:
        conf.update(json.load(jsonfile))

    conf["EXEC_TIME"] = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    # Initialize the different paths from the configuration loaded previously
    init_path(conf['DIRECTORIES']['DATA_DIRECTORY'])





def init_path(data_path):
    """
    The init_path function initializes a dictionary of paths for different data directories based on
    the provided data_path. It adjusts the path format for Windows systems if necessary

    Parameters
    ----------
    data_path: str
        The base directory path where data directories are located

    """
    if conf["SYSTEM"] == "WIN32" and 'OMA' in data_path:
        data_path = data_path[1:]
    conf["PATH"] = {
        'REFERENCES': os.path.join(data_path, "00_references"),
        'RAW_DATA': os.path.join(data_path, "01_raw"),
        'INTERMEDIATE_DATA': os.path.join(data_path, "02_intermediate"),
        'PROCESSED_DATA': os.path.join(data_path, "03_processed"),
        'MODELS': os.path.join(data_path, "04_models"),
        'MODELS_OUTPUT': os.path.join(data_path, "05_models_output"),
        'RANDIM': os.path.join(data_path, "06_randim"),
        'FINAL': os.path.join(data_path, "07_final_outputs"),
        'CAPACITY': os.path.join(data_path, '08_capacity')
    }
