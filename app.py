import logging, json
import sys
import traceback
from datetime import date
from logging.handlers import RotatingFileHandler
from flask import Flask, request, Response
import config.api_config as api_config
import src.validation as standard_validation
import src.oss_validation as oss_validation
from src.purge import purge
import sqlalchemy
import pandas as pd
import os, io
from google.cloud import bigquery, storage
from google.oauth2 import service_account

#quick commit

app = Flask('main_app')

def create_ifne(folder_path):
                # Create folder if it does not exist
                if not os.path.exists(folder_path): os.makedirs(folder_path)

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s %(levelname)-8s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    handlers=[RotatingFileHandler(api_config.log_file, maxBytes=api_config.log_max_byte,
                                                  backupCount=api_config.log_backup_count),
                              logging.StreamHandler(sys.stdout)]
                    )

@app.route('/validation', methods=['POST'])
def validate():
    try:
        logging.info("Request POST /")
        posted_json = request.get_json()

        file_type = posted_json['file_type']
        mode = posted_json['mode'].lower() if 'mode' in posted_json else 'local'
        delimiter = posted_json['delimiter'] if 'delimiter' in posted_json else ','
        join_paths = posted_json['join_paths'] if 'join_paths' in posted_json else {}
        n_examples = posted_json['n_examples'] if 'n_examples' in posted_json else 5
        init = posted_json['init'] if 'init' in posted_json else False
        sql = posted_json['sql'] if 'sql' in posted_json else "False"
        type_class_map = {}
        

        if mode in ['bk','bq']:
            # Read the type class map from GCP Storage
            # Client
            #cred_json = r'sa-ocdvt-key.json'
            #project_id = posted_json['project_id'] if 'project_id' in posted_json else "ino-dataai-data2-gov2-2021-sbx"
            #credentials = service_account.Credentials.from_service_account_file(cred_json)
            #client = storage.Client(credentials= credentials,project=project_id)
            client = storage.Client()
            # Read Type Class Map
            bucket_name = posted_json['bucket_name'] if 'bucket_name' in posted_json else 'bucket-ocdvt'
            data = json.loads(client.get_bucket(bucket_name).blob(r'schema/type_class_map.json').download_as_string())
        
            # Reading Schema
            schema = json.loads(client.get_bucket(bucket_name).blob(r'schema/%s/%s_schema.json'%(file_type, file_type)).download_as_string())
            with open(r'config/%s_schema.json'%(file_type), 'w') as outfile:
                json.dump(schema, outfile)
        
        else:
            # Read the type class map from local folder
            file = open("type_class_map.json", 'r')
            data = json.load(file)
        
        for key, value in data.items():
            if value == 'oss_validation.OssValidation':
                type_class_map[key] = oss_validation.OssValidation
            else:
                type_class_map[key] = standard_validation.Validation
        #print(type_class_map)
        
        # Validation Process
        validation_class = type_class_map[file_type]
        validation_tool = validation_class(file_type, join_paths, n_examples)

        if mode.lower() == 'bq':
            # Arguments
            project_id = posted_json['project_id'] if 'project_id' in posted_json else "ino-dataai-data2-gov2-2021-sbx"
            bigquery_dataset_name = posted_json['bigquery_dataset_name'] if 'bigquery_dataset_name' in posted_json else 'bq_ocdvt'
            table_name = posted_json['table_name'] if 'table_name' in posted_json else 'oss_tuto'

            # Create connection to GBQ
            #credentials = service_account.Credentials.from_service_account_file(cred_json)
            #client = bigquery.Client(credentials= credentials, project=project_id)
            client =  bigquery.Client()
            sql_query = "SELECT * FROM %s.%s"%(bigquery_dataset_name, table_name)
            data = client.query(sql_query).result().to_dataframe()
            
            if init:
                validation_tool.init()
            
            anomalies = validation_tool.validate_data(data = data, data_delimiter = delimiter)
            examples, anomalies = validation_tool.get_anomalies_examples(anomalies, data=data, data_delimiter = delimiter)

            # Cast examples to str because there is no list type in mysql
            anomalies.examples = anomalies.examples.map(str)
            # reset index because text pk raise exception by MySql
            anomalies = anomalies.reset_index()
            
            # Export results to BQ Table
            job_config1 = bigquery.job.LoadJobConfig()
            job_config1.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            
            table_id = '%s.%s.%s'%(project_id, bigquery_dataset_name, "examples")
            client.load_table_from_dataframe(examples, table_id, job_config = job_config1).result()

            
            job_config2 = bigquery.job.LoadJobConfig()
            job_config2.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE  

            table_id = '%s.%s.%s'%(project_id, bigquery_dataset_name, "anomalies")
            client.load_table_from_dataframe(anomalies, table_id, job_config = job_config2).result()

        elif mode.lower() == 'bk':
            
            # Arguments
            csv_path = r'data/%s/%s.csv'%(file_type, file_type)

            # Read CSV from bucket
            data = client.get_bucket(bucket_name).blob(csv_path).download_as_bytes()
            data = pd.read_csv(io.BytesIO(data), sep = delimiter)

            if init:
                validation_tool.init()
            
            anomalies = validation_tool.validate_data(data = data, data_delimiter=';')
            examples, anomalies = validation_tool.get_anomalies_examples(anomalies, data=data, data_delimiter = delimiter)

            # Cast examples to str because there is no list type in mysql
            anomalies.examples = anomalies.examples.map(str)
            # reset index because text pk raise exception by MySql
            anomalies = anomalies.reset_index()

            # Export results to CSV
            client.get_bucket(bucket_name).blob('results/%s/anomalies.csv'%(file_type)).upload_from_string(anomalies.to_csv(index=False),'text/csv')
            client.get_bucket(bucket_name).blob('results/%s/examples.csv'%(file_type)).upload_from_string(examples.to_csv(index=False),'text/csv')

        elif mode.lower() == 'local':
            
            dq_folder_path = posted_json['dq_folder_path'] if 'dq_folder_path' in posted_json else 'dq_report/'
            create_ifne(dq_folder_path)
            path = posted_json['path'] if 'path' in posted_json else 'train.csv'

            if init:
                validation_tool.init()

            anomalies = validation_tool.validate_data(data_path=path, data_delimiter=delimiter)
            
            examples, anomalies = validation_tool.get_anomalies_examples(anomalies, data_path=path, data_delimiter=delimiter)

            # Cast examples to str because there is no list type in mysql
            anomalies.examples = anomalies.examples.map(str)
            # reset index because text pk raise exception by MySql
            anomalies = anomalies.reset_index()
            print(anomalies)

        else:
            file_type = posted_json['file_type']
            path = posted_json['path'] if 'path' in posted_json else ''
            delimiter = posted_json['delimiter'] if 'delimiter' in posted_json else ','
            previous_data_path = posted_json['previous_data_path'] if 'previous_data_path' in posted_json else None
            previous_data_delimiter = posted_json['previous_data_delimiter'] if 'previous_data_delimiter' in posted_json else ','
            exclude = posted_json['exclude'] if 'exclude' in posted_json else []

            if init:
                validation_tool.init()

            anomalies = validation_tool.validate_data(data_path=path, data_delimiter=delimiter, exclude=exclude,
                                                  previous_data_path=previous_data_path,
                                                  previous_data_delimiter=previous_data_delimiter)
            
            examples, anomalies = validation_tool.get_anomalies_examples(anomalies, data_path=path, data_delimiter=delimiter)

            # Cast examples to str because there is no list type in mysql
            anomalies.examples = anomalies.examples.map(str)
            # reset index because text pk raise exception by MySql
            anomalies = anomalies.reset_index()
    
        if sql == "True":    
            #SQL Connection
            logging.info("Open validation db connection")
            # Db Connection 
            validation_engine = sqlalchemy.create_engine(api_config.url,
                                                     connect_args={'host': api_config.validation_host,
                                                                   'database': api_config.validation_database,
                                                                   'user': api_config.validation_user,
                                                                   'password': api_config.validation_password
                                                                   }
                                                     )

            logging.info("Push to db")

            anomalies.to_sql(api_config.anomalies_table,
                         con=validation_engine,
                         if_exists='append',
                         index=False
                         )

            examples.to_sql(api_config.examples_table,
                        con=validation_engine,
                        if_exists='append',
                        index=False
                        )

            logging.info("Open supervision db connection")
            # Db Connection
            supervision_engine = sqlalchemy.create_engine(api_config.url,
                                                      connect_args={'host': api_config.supervision_host,
                                                                    'database': api_config.supervision_database,
                                                                    'user': api_config.supervision_user,
                                                                    'password': api_config.supervision_password
                                                                    }
                                                      )
            logging.info("Push to db")
            supervision_log = pd.DataFrame({'date': [str(date.today())],
                                        'projet': [api_config.supervision_project],
                                        'job': [api_config.supervision_job + ' :' + file_type],
                                        "statut": ['SUCCESS']
                                        })

            supervision_log.to_sql(api_config.supervision_table,
                               con=supervision_engine,
                               if_exists='append',
                               index=False
                               )
        
        logging.info("Process finished")
        return Response(status=200)

    except Exception as e:
        app.logger.error(traceback.format_exc())
        try :
            if sql == "True":    
                logging.info("Push to db")
                supervision_log = pd.DataFrame({'date': [str(date.today())],
                                            'projet': [api_config.supervision_project],
                                            'job': [api_config.supervision_job + ' :' + file_type],
                                            "statut": ['FAILURE']
                                            })
                supervision_log.to_sql(api_config.supervision_table,
                                   con=supervision_engine,
                                   if_exists='append',
                                   index=False
                                   )
                logging.info("Process finished")
        except Exception as e:
            if sql =='True':
                app.logger.error(traceback.format_exc())
                app.logger.error("can't save to supervision table")
        return Response(status=500)


@app.route('/purge')
def purge_data():
    try:
        logging.info("Purge start")
        posted_json = request.get_json()
        logs = posted_json['logs'] if 'logs' in posted_json else False
        history = posted_json['history'] if 'history' in posted_json else False
        db = posted_json['db'] if 'db' in posted_json else False
        purge(logs, history, db)
        return Response(status=200)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return Response(status=500)


if __name__ == '__main__':
    app.run(host=api_config.flask_host, port=api_config.flask_port)