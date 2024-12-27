import logging
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


import os
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq as gbq

app = Flask('main_app')

type_class_map = {
    'oss': oss_validation.OssValidation,
    'oss2g': oss_validation.OssValidation,
    'oss3g': oss_validation.OssValidation,
    'oss4g': oss_validation.OssValidation,
    'cdr': standard_validation.Validation,
    'cp': standard_validation.Validation,
    'cells': standard_validation.Validation,
    'sites': standard_validation.Validation,
    'cgi_mapping': standard_validation.Validation,
    'frame3g': oss_validation.OssValidation,
    'cong2g': oss_validation.OssValidation,
    'oss_tuto' : oss_validation.OssValidation
}
logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s %(levelname)-8s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    handlers=[RotatingFileHandler(api_config.log_file, maxBytes=api_config.log_max_byte,
                                                  backupCount=api_config.log_backup_count),
                              logging.StreamHandler(sys.stdout)]
                    )

def query_to_df(client, sql_query):
    query_job = client.query(sql_query)
    results = query_job.result()
    df = (query_job.result().to_dataframe())
    return df

@app.route('/')
def validate():
    try:
        logging.info("Request get /")
        posted_json = request.get_json()
        #
        cred_json = r'ino-dataai-data2-gov2-2021-sbx-6672be98f585.json'
        credentials = service_account.Credentials.from_service_account_file(cred_json)
        project_id =  "ino-dataai-data2-gov2-2021-sbx"
        bigquery_dataset_name = 'kli_ds_test'
        table_name = 'preprocessed'
        client = bigquery.Client(credentials= credentials,project=project_id)
        
        file_type = posted_json['file_type']
        path = posted_json['path']
        delimiter = posted_json['delimiter'] if 'delimiter' in posted_json else ','
        previous_data_path = posted_json['previous_data_path'] if 'previous_data_path' in posted_json else None
        previous_data_delimiter = posted_json['previous_data_delimiter'] if 'previous_data_delimiter' \
                                                                            in posted_json else ','
        init = posted_json['init'] if 'init' in posted_json else False
        join_paths = posted_json['join_paths'] if 'join_paths' in posted_json else {}
        n_examples = posted_json['n_examples'] if 'n_examples' in posted_json else 5
        exclude = posted_json['exclude'] if 'exclude' in posted_json else []

        validation_class = type_class_map[file_type]
        validation_tool = validation_class(file_type, join_paths, n_examples)

        if init:
            validation_tool.init()

        #anomalies = validation_tool.validate_data(data_path=path, data_delimiter=delimiter, exclude=exclude,
        #                                          previous_data_path=previous_data_path,
        #                                          previous_data_delimiter=previous_data_delimiter)
        
        sql_query = "SELECT *FROM %s.%s"%(bigquery_dataset_name, table_name)

        data = query_to_df(client, sql_query)

        anomalies = validation_tool.validate_data(data = data, data_delimiter=';')

        examples, anomalies = validation_tool.get_anomalies_examples(anomalies, data=data, data_delimiter=';')

        # Cast examples to str because there is no list type in mysql
        anomalies.examples = anomalies.examples.map(str)
        # reset index because text pk raise exception by MySql
        anomalies = anomalies.reset_index()

        # Export to BQ Table
        gbq.to_gbq(examples, destination_table ="%s.%s"%(bigquery_dataset_name,"examples"),project_id=project_id, if_exists = 'replace', credentials = credentials)
        gbq.to_gbq(anomalies, destination_table ="%s.%s"%(bigquery_dataset_name,'anomalies'),project_id=project_id, if_exists = 'replace', credentials = credentials)


        #SQL Connection
        #logging.info("Open validation db connection")
        ## Db Connection 
        #validation_engine = sqlalchemy.create_engine(api_config.url,
        #                                             connect_args={'host': api_config.validation_host,
        #                                                           'database': api_config.validation_database,
        #                                                           'user': api_config.validation_user,
        #                                                           'password': api_config.validation_password
        #                                                           }
        #                                             )
#
        #logging.info("Push to db")
#
        #anomalies.to_sql(api_config.anomalies_table,
        #                 con=validation_engine,
        #                 if_exists='append',
        #                 index=False
        #                 )
#
        #examples.to_sql(api_config.examples_table,
        #                con=validation_engine,
        #                if_exists='append',
        #                index=False
        #                )
#
        #logging.info("Open supervision db connection")
        ## Db Connection
        #supervision_engine = sqlalchemy.create_engine(api_config.url,
        #                                              connect_args={'host': api_config.supervision_host,
        #                                                            'database': api_config.supervision_database,
        #                                                            'user': api_config.supervision_user,
        #                                                            'password': api_config.supervision_password
        #                                                            }
        #                                              )
        #logging.info("Push to db")
        #supervision_log = pd.DataFrame({'date': [str(date.today())],
        #                                'projet': [api_config.supervision_project],
        #                                'job': [api_config.supervision_job + ' :' + file_type],
        #                                "statut": ['SUCCESS']
        #                                })
#
        #supervision_log.to_sql(api_config.supervision_table,
        #                       con=supervision_engine,
        #                       if_exists='append',
        #                       index=False
        #                       )
        #logging.info("Process finished")
        #return Response(status=200)

    except Exception as e:
        app.logger.error(traceback.format_exc())
        try :
            logging.info("Push to db")
            #supervision_log = pd.DataFrame({'date': [str(date.today())],
            #                                'projet': [api_config.supervision_project],
            #                                'job': [api_config.supervision_job + ' :' + file_type],
            #                                "statut": ['FAILURE']
            #                                })
            #supervision_log.to_sql(api_config.supervision_table,
            #                       con=supervision_engine,
            #                       if_exists='append',
            #                       index=False
            #                       )
            #logging.info("Process finished")
        except Exception as e:
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