## config.py : 
a python module with the paths used by
* src/validation.py:
  - previous_data : csv that contains the last validated data
  - schema_path = path to the schema
  
* src/oss_validation.py: 
  - cells_list : list of previous batch's cells name
  - time_series_history = a dict with a summarized history of the counters per cell
  - history_folder : path to the folder where 'cells_list' and 'time_series_history' are saved

## api_config.py : 
a python module with the paths used by
* Flask
  - flask_host = the host's ip
  - flask_port = port to sue by flask
  
* Logs 
  - log_file = file where to save logs
  - log_max_byte = max log file size
  - log_backup_count = number of backup files before rollover

* MySql  
  - table_name = the table used to save anomalies reports
  - url = db url prefix "mysql+pymysql://"
  - host = the host's ip
  - user = the username
  - password = the password
  - database = the name of the database

Nb: it is recommended to have a shared volume for each of the history and logs folders in order to keep them if the 
container crashes