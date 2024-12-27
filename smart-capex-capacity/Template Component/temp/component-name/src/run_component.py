

from google.cloud import bigquery
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
from .mycode import myfunction # import your module

# import parameters
parser = argparse.ArgumentParser()
parser.add_argument('--PROJECT_ID', dest = 'PROJECT_ID', type = str)
parser.add_argument('--DATANAME', dest = 'DATANAME', type = str)
parser.add_argument('--NOTEBOOK', dest = 'NOTEBOOK', type = str)

parser.add_argument('--my_arg', dest = 'my_arg', type = str) # all all your arguments


args = parser.parse_args()
PROJECT_ID = args.PROJECT_ID
DATANAME = args.DATANAME
NOTEBOOK = args.NOTEBOOK

my_arg = args.my_arg
print(PROJECT_ID, DATANAME, NOTEBOOK)

# client for BQ
bq = bigquery.Client(project = PROJECT_ID)

query = f"SELECT * FROM `{PROJECT_ID}.{DATANAME}.source` ORDER by cell_name, date" 

source = bq.query(query = query).to_dataframe()

output = myfunction(my_arg)

# output data - to BQ
output.to_gbq(f"{PROJECT_ID}.{DATANAME}.{NOTEBOOK}", f'{PROJECT_ID}', if_exists = 'replace')

