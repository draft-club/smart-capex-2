# This is mirrored to the repo : https://source.cloud.google.com/ino-dataai-data2-gov2-2021-sbx/ocdvt-mirror
# Data validation solution for SmartCAPEX : 

# Documentation about cloud build
https://cloud.google.com/build/docs/deploying-builds/deploy-cloud-run?hl=fr
# Documentation about cloud run arguments
https://cloud.google.com/sdk/gcloud/reference/run/deploy

### set db connection :
After cloning the project, the first thing to do is to open the config/api_config.py file and set the empty fields to set up the db connection

### Build docker image : 

`docker build -t <tag> --build-arg country="<country>" .`

Or with proxy:

`docker build -t <tag>  --build-arg http_proxy="http://<host>:<port>" --build-arg https_proxy="<host>:<port>" --build-arg country="<country>"  .`

Nb: country could be either obf or oci
  

### Run Docker container :

`docker run --rm -d -v "<path_to_shared_data>":"<mounting_destination>" -v "<path_to_history>":"/history" 
-v "<path_to_logs>":"/logs" -p <host_port>:<container_port> --name <container-name> <image-tag>`

### consuming the api : 

`requests.get('<host>:<port>', json={'path':<path_to_csv_to_validate>,'delimiter':<csv_delimiter>,
'previous_data_path':<path_to_csv_to_validate>,'previous_data_delimiter':<csv_delimiter>,
'file_type': <file_type>, n_examples:<number_of_anomalies_example_to_return>, exclude:<list_of_features_to_exclude>,
'init':<whether to init history>, 'join_paths':{<join_name_1>: {'path': <path_1>, 'delimiter': <del_1>},
<join_name_2>: {'path': <path_2>, 'delimiter': <del_2>}}})`

Nb: 
* data path in the request should be the path from the docker container's root
* join_name should match the name of the right table as mentioned in the schema file
* supported file_type are : oss, oss2g, oss3g, oss4g, frame3g, cong2g, cdr, cp, cells, sites, cgi_mapping. 
* Any other type of validation could be added to the standard validation package by :
  - creating another validation class that inherits from the validation class (e.g. OssValidation)
  - adding to this class the methods executing new type of validations, these function should return a list with the following format : `[[<feature name>, <anomaly key>, <anomaly message>, <examples>]]`
  - override _validate_custom_constraints to include the list resulting from the new validations in its return df
  - update type_class_map dict in app.py if in prod or use 'add_type_class_matching' in runner.py to update it if dev
  

- in order to purge all the app data including the db tables, a request to the following URL should be made <br>
`requests.get('http://10.238.36.21:5000/purge', json={'logs':<bool>, 'history':<bool>, 'db':<bool>})` <br>
- the scripts folder contains script to request each of the two URLS seen above


# Data validation: Functionalities
### Goal: validate data used by Smart Capex to detect anomalies
- List of dataset covered:
    - oss-counters
    - customer portfolio
    - site dictionary
    - CDRs dictionary
    - cells
- Input for validation: 
    - the current batch of data 
    - the previous batch of data (data from the previous week)
    - a set of parameters (json config file)
    - output: an anomaly report saved in a table in the database

### List of validations implemented
- Global statistics from one week to previous one
    - Nb of records of current week


- Presence in a dataset compared to previous week (for oss-counter datasets)
    - No more than 300 new cells  
    - No more than 200 missing cells


- Columns validations 
    - Unicity
    - Distinctness (nb of distinct values / nb of records)
    - Type check
    - Value check (or domain for categorical data)
    - Format (including regex)
    - presence
    - drift in distribution
    - outlier detection
    - highest frequency threshold
    - mapped to only one
    - joins validations

### Schema tools
the package schema_tools in src contain the necessary functions to create and manipulate schemas used in this project, 
this package offers the possibility to get, set and clear each attribute of the schema.

### Interfaces and API
- data loading
    - files are read from Smart Capex system file storage 
- result 
    - result of anomaly detection is stored in a table inside Smart capex database

# Technical architecture
The data validation module is delivered as a Flask API that runs on a docker container.

