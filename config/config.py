import os
from pathlib import Path
current_path = str(os.path.abspath(__file__))
path = Path(current_path)
conf_dir_path = str(path.parent.absolute())

# oss args
history_folder = '../history'
cells_list = history_folder + '/{}_cells.txt'
time_series_history = history_folder + '/{}_time_series.json'
#previous_data = history_folder + '/previous_{}.csv'
previous_data = "dq_report" + '/previous_{}.csv'
schema_path = conf_dir_path + '/{}_schema.json'
