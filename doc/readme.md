# Data Quality for SmartCapex:
## Description :
As the vast majority of AI projects nowadays, SmartCapex is based on Machine Learning algorithms. To exercise their 
predictive power, they largely depend on data. As a result, their quality strongly influences the relevance of these *
solutions. However, obtaining and maintaining high quality data is not always easy. Many data quality factors threaten 
to derail AI programs.<br>
The objective of our Data quality for SmartCapex solution is to detect quality flaws in the data used by SmartCapex in 
order to give the user the possibility of intervening to carry out the necessary actions and avoid the "garbage-in 
garbage-out" effect.
## Approach:
Our solution is based on a key element "schema". it is a json file that encode data constraints, it could be 
generated and adjusted. The validation process is simply to compare the data to the constraints encoded in the 
schema.<br>
## Use modes:
We give the user the possibility to use the solution in two modes :
- Development mode (Exploratory): the solution can be used in a handy way by importing runner.py module into a
  jupyter notebook. this module affords all the functionalities of the solution. Practical in development phase with 
  need of being able to validate data in demand in order to perform tests and when the schema is frequently modified.
- Production mode (Automatic): a user can use the data validation api to validate data. Practical in production where 
  the user have a consistent flow of data to validate.

## Repository structure 
<img width="600" src="Capture 1.PNG"/><br>
SmartCapex-dv repository contains the following folders:
- config: A folder with the config files that manage the connection with DBs, API deployment, logs and history locations
  etc. before using the solution, it is mandatory to put the schemas that we are going to need in that folder.
- notebooks: Contain the notebooks used to analyse the data from each country. Additionally, it contains the runner.py 
  that allows the use of the solution in a notebook
- schema_registry: Contains schemas for the data types used by SmartCAPEX for each country.
- scripts: Contains example of http request using python, the docker build script and the crontab configuration script
- src : A folder with the solution modules 
- test : Contains unit test for the validation modules
 
## Supported file types:
- **OCI** :  cdr, cp , oss, cells, sites, cong2g, frame3g, oss3g, oss4g
- **ORDC** : oss2g, oss3g, oss4g, sites, cells, cgi_mapping
- **OBF** : cdr, cells, oss, sites
NB: cp stands for costumer portfolio 

## Workflow:
##### Exploratory:
<img width="600" src="Capture 2.PNG"/><br>
In the Exploratory mode, we use the schema_tools module to generate a first version of the schema. The latter is 
adjusted to include are the constraints that we want our future data to respect. Then we can proceed to validation and 
obtain an anomalies report as a df
##### Automatic:
<img width="600" src="Capture 3.PNG"/><br>
In this mode we use already prepared schemas. The api receive the http request and use the suited validation script 
and the suited schema for the concerned file type. The result of the validation are pushed to a DB 

## Schema:
The schema is the structure in which we encode our constraint for a given data type, the solution offers 4 categories of
constraints:
### dataset_constraint : Constraint for the whole dataset at once
- min_fraction_threshold : min fraction of the number of actual dataset length compared to previous dataset
- max_fraction_threshold : max fraction of the number of actual dataset length compared to previous dataset
- unicity_features : features used to check for duplicates. could be a list of features or `"*"` to use all the features
- batch_id_column : the column that contain the id of the batch to validate
- joins_constraint : list of dict with 4 keys
  - name : name of the right join table
  - left_on : feature (str) or features ('list') from the left join table
  - right_on : feature (str) or features ('list') from the right join table
  - threshold : the min possible value for the ratio of the size of the join table on the original one

### features_constraint : Constraint related to a given feature
- name : name of the concerned feature
- type : type of the feature : int, float or str
- presence : fraction [0,1] of values that should be present within a feature
- distinctness : a condition of distinctness measure that indicates the ratio of unique values in a feature. it is 
  calculated as follows: f(x,y) = 0 if x == 1 else x/y; with x the number of unique values and y the number of 
  non-missing values. defined in the schema as a tuple of { "condition" : "" , "value" : "" }
    - condition : one of the values "eq" for equal to, "lt" for less than and "gt" for greater than.
    - value : value  [0,1] used with the condition 
<br><br>
- domain : list/interval of possible values that a feature could take. Should conform feature type. it could be 
  defined as a list of values or as a max and/or min attribute for numerical features.
- regex : pattern that the feature should match. in addition to regular regex special characters, we could use # to 
  refer to another feature<br>
  example : "^#year#(0)?#week#$" the #year# would be replaced with the value of the year column before applying the 
  regex<br>
  It is also possible to add only some parts of a column <br>
  example : "^#year#(0)#domain[2,4]#$" will use the full value of the column year, a regex and the 2nd to 4th characters
  of the column domain<br>
- drift : a threshold for Tchebychev distance between the current batch and the previous one. the previous batch is
  found in a history folder.
- outliers : a float value k that define the number of stds around the means that define the non outlier domain <br>
  non outlier domain : [mean - k*std , mean + k*std] <br>
- highest_frequency_threshold : the max frequency that a value could have within column. it could be an integer (max 
  allowed number of repetitions) or a float [0,1] (the ratio within non-missing values) 
- mapped_to_only_one : the feature or the features that should always take the same value for a given value of the 
  concerned feature 


### slices_constraint : to define constraints for a slice of the data
in this entry we should define : 
- slicing column : column on which we are going to slice 
- values to take : array. keep the rows where slicing column is in values to take
- values to drop : array. drop the rows where slicing column is in values to drop
- schema : an embedded schema with all the constraints to validate for the resulting slice of data.

### custom_constraint:
Contains all the entries that user want to add in order to perform custom validations. For OSS, we have the following 
entries:
- max_new_cells : maximum number of allowed new cells compared to previous dataset
- max_disappeared cells : maximum number of allowed disappeared cells compared to previous dataset
- max_cell_per_site : maximum allowed number of cells per site
- cell_name_feature : the feature that contains the cell name (to detect outliers using time series history)

## Modules:
### Validation Modules:
Two validation classes have been developed:
- Validation: in validation.py. a generic class that contains the function to perform all the validations for the 
  constraints mentioned in the schema. with this class we can validate all file type except oss, hence the need of 
  another validation class.
- OssValidation: in OssValidation.py. The outlier detection function used for the other file types is not suitable for 
  oss counters. For those feature what matters is the history of the counter detected by a given cell and not the values
  of other cells in the same batch. Additionally, there is some other validations that we want to perform and that are 
  very specific to the oss context.<br> 
  The solution was to create a class that inherits from validation, that include the additional validations and that 
  override the outlier detection function.<br>
  The history of the counters for each cell and the list of the cells are saved on a history folder
### Schema tools Module:
The package schema_tools in src contain the necessary functions to create and manipulate schemas used in this project, 
this package offers the possibility to get, set and clear each attribute of the schema.
### Runner Module:
The module that we use in dev mode. it allows us to use the solution in a notebook.

## Config folder:
A folder with the config files that manage the connection with DBs, API deployment, logs and history locations etc. 
before using the solution, it is mandatory to put the schemas that we are going to need in that folder. it contains two 
configuration files.
#### config.py : 
a python module with the paths used by
* src/validation.py:
  - previous_data : csv that contains the last validated data
  - schema_path = path to the schemas
  
* src/oss_validation.py: 
  - cells_list : list of previous batch's cells name
  - time_series_history = a dict with a summarized history of the counters per cell
  - history_folder : path to the folder where 'cells_list' and 'time_series_history' are saved

#### api_config.py : 
a python module with the paths used by
* Flask
  - flask_host = the host's ip
  - flask_port = port to use by flask
  
* Logs 
  - log_file = file where to save logs
  - log_max_byte = max log file size
  - log_backup_count = number of backup files before rollover

* Reports db config   
  - anomalies_table = the table used to save anomalies reports
  - examples_table = the table used to save anomalies examples
  - url = db url prefix "mysql+pymysql://"
  - validation_host = the host's ip
  - validation_user = the username
  - validation_password = the password
  - validation_database = the name of the database

* Supervision db config   
  - supervision_table = the table used to save anomalies reports
  - url = db url prefix "mysql+pymysql://"
  - supervision_host = the host's ip
  - supervision_user = the username
  - supervision_password = the password
  - supervision_database = the name of the database
  - supervision_project = the name of the project
  - supervision_job = the name of the job

NB: it is recommended to have a shared volume for each of the history and logs folders in order to keep them if the 
container crashes

## Deployment
### Technical architecture
The data validation module is delivered as a Flask API that runs on a docker container.
### Set db connections :
After cloning the project, the first thing to do is to open the config/api_config.py file and set the empty fields 
to set up the db connections

### Build docker image : 

`docker build -t <tag> --build-arg country="<country>" .`

Or with proxy:

`docker build -t <tag>  --build-arg http_proxy="http://<host>:<port>" --build-arg https_proxy="<host>:<port>" 
 --build-arg country="<country>"  .`

Nb: country could be either oci, obf or ordc
  

### Run Docker container :

`docker run --rm -d -v "<path_to_shared_data>":"<mounting_destination>" -v "<path_to_history>":"/history" 
 -v "<path_to_logs>":"/logs" -p <host_port>:<container_port> --name <container-name> <image-tag>`

### Consuming the api : 
#### Validation:
`requests.get('<host>:<port>', json={'path':<path_to_csv_to_validate>,'delimiter':<csv_delimiter>,
'previous_data_path':<path_to_csv_to_validate>,'previous_data_delimiter':<csv_delimiter>,
'file_type': <file_type>, n_examples:<number_of_anomalies_example_to_return>, exclude:<list_of_features_to_exclude>,
'init':<whether to init history>, 'join_paths':{<join_name_1>: {'path': <path_1>, 'delimiter': <del_1>},
<join_name_2>: {'path': <path_2>, 'delimiter': <del_2>}}})`

Nb: 
* data path in the request should be the path from the docker container's root
* join_name should match the name of the right table as mentioned in the schema file
* supported file_type are : oss, oss2g, oss3g, oss4g, cdr, cp, cells, sites, cgi_mapping. 

#### Purge : 
in order to purge all the app data including the db tables, a request to the following URL should be made <br>
`requests.get('http://10.238.36.21:5000/purge', json={'logs':<bool>, 'history':<bool>, 'db':<bool>})`

#### Examples:
Scripts folder contains scripts to request each of the two URLS seen above

## Validation results [WIP]: 
The Anomaly reports are published in a dashboard<br>
<img width="600" src="Capture 4.PNG"/><br>

## Support other file types:
the solution is easily extendable to support other file types. Here we have two cases:
### Validation script is enough to validate the new file type:
The steps here are as follows : 
- update type_class_map dict in app.py if in prod or use 'add_type_class_matching' in runner.py to update it if dev<br>
**Example** : `type_class_map = {
    'new_type': standard_validation.Validation,
    'oss': oss_validation.OssValidation,
    'cgi_mapping': standard_validation.Validation
    ...
}`
### New file type needs other validations or a modified version of existing validations:
The steps here as follows : 
- Create another validation class that inherits from the validation class (e.g. OssValidation)
- Add to this class the methods executing new type of validations, these methods should return a list with the 
  following format : `[[<feature name>, <anomaly key>, <anomaly message>, <examples>]]`
- Override _validate_custom_constraints to include the list resulting from the new validations in its return df
- Update type_class_map dict in app.py if in prod or use 'add_type_class_matching' in runner.py to update it if dev