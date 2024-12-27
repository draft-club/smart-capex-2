import logging
import sqlalchemy
import config.api_config as api_config
import os
import config.config as config


def purge(logs: bool, history: bool, db: bool):
    """
    a function to erase module data
    logs: whether to erase logs
    history: whether to erase history
    db: whether to erase db tables
    """
    if logs:
        log_path = api_config.log_file
        log_folder = log_path[:log_path.rfind('/')]
        files = os.listdir(log_folder)
        for file in files:
            logging.info("removing "+log_folder + "/" + file)
            os.remove(log_folder + "/" + file)

    if history:
        history_folder = config.history_folder
        files = os.listdir(history_folder)
        for file in files:
            logging.info("removing "+history_folder + "/" + file)
            os.remove(history_folder + "/" + file)

    if db:
        logging.info("removing db tables")
        engine = sqlalchemy.create_engine(api_config.url,
                                          connect_args={'host': api_config.host,
                                                        'database': api_config.database,
                                                        'user': api_config.user,
                                                        'password': api_config.password
                                                        }
                                          )
        connection = engine.raw_connection()
        cursor = connection.cursor()
        command = "drop table if exists {};"
        cursor.execute(command.format(api_config.anomalies_table))
        cursor.execute(command.format(api_config.examples_table))
        connection.commit()
        cursor.close()
