import getpass
from pathlib import Path
import os
import datetime
import json



exec_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
conf = {}


def conf_loader(country):
    """
    Load configuration from country specific json file and intialize paths
    """

    with open(f"src/{country.lower()}.json") as jsonfile:
        global conf
        conf.update(json.load(jsonfile))

    conf["EXEC_TIME"] = exec_time


