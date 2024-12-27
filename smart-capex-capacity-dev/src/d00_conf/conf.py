import getpass
from pathlib import Path
import os
import datetime
import json

# Start to record execution time
exec_time = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

# Get current user
user = getpass.getuser()

# Variable used in debug to save data in local home
SAVE_IN_LOCAL_PATH = False

# Create an empty dictionary containing the global configuration
# Loaded in main from ****.json file
conf = {}

def conf_loader(country):
    """
    Load configuration from country specific json file and intialize paths
    """
    # with open(f'./d00_conf/{country.lower()}.json') as jsonfile:
    with open(f'src/d00_conf/{country.lower()}.json',encoding='utf8') as jsonfile: # src/ in capacity
        global conf
        conf.update(json.load(jsonfile))

    conf["EXEC_TIME"] = exec_time
    # Initialize the different paths from the configuration loaded previously
    init_path()




def init_base_path():
    """
    Initialize the base path according to local or production mode
    """
    # Local debug mode
    if SAVE_IN_LOCAL_PATH:
        """ Project specific constants"""
        if (user == 'perezfernando'):
            BASE_PATH = "/Users/perezfernando/Documents/smartcapex_oci_repo"
        elif (user == 'Marc Olle'):
            BASE_PATH = "marc.ole"
        elif (user == 'ivan.sidorenko') :
            BASE_PATH = "/home/ivan.sidorenko/SmartCapex"
        elif (user == 'damien.rousseau'):
            BASE_PATH = "/home/damien.rousseau/smartcapex"
        else:
            BASE_PATH = "/home/ubuntu/smartcapex"
    #Production mode
    else:
        BASE_PATH = '/'

    return BASE_PATH


def init_path():
    """
    Initialize all the path
    """
    # Init the base path
    BASE_PATH = init_base_path()

    # Init the data path: by default we manage OCI data
    DATA_PATH = '/data'
    if conf['COUNTRY'] != 'OCI':
        DATA_PATH = f"/data/{conf['COUNTRY']}"



    conf["PATH"] = {
        'BASE': BASE_PATH,
        'REFERENCES': os.path.join(DATA_PATH, "00_references"),
        'RAW_DATA': os.path.join(DATA_PATH, "01_raw"),
        'INTERMEDIATE_DATA': os.path.join(DATA_PATH, "02_intermediate"),
        'PROCESSED_DATA': os.path.join(DATA_PATH, "03_processed"),
        'MODELS': os.path.join(DATA_PATH, "04_models"),
        'MODELS_OUTPUT': os.path.join(DATA_PATH, "05_models_output"),
        'REPORTING': os.path.join(DATA_PATH, "06_reporting")
    }

