import logging
import sys
from pathlib import Path
import os

current_path = str(os.path.abspath(__file__))
path = Path(current_path)
root_dir = str(path.parent.parent.absolute())
sys.path.insert(0, root_dir)

import pandas as pd
import src.validation as standard_validation
import src.oss_validation as oss_validation

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
    'default': standard_validation.Validation,
    'frame3g': oss_validation.OssValidation,
    'cong2g': oss_validation.OssValidation,
    'oss3g': oss_validation.OssValidation,
    'oss4g': oss_validation.OssValidation,
}


def validate(file_type: str, data: pd.DataFrame = None, path: str = None, data_delimiter: str = ',', join_paths=None,
             init: bool = False, exclude: list = None, n_examples: int = 5, dq_folder_path = 'dq_report/') -> pd.DataFrame:
    """
    a function that calls the validation module to validate a given dataset
    :param file_type: could be either oss, cdr, cp, cells or sites
    :param data: df with the dataset to validate
    :param path: path to csv file that contains the dataset to validate
    :param data_delimiter: delimiter of the csv file
    :param join_paths: a dict that contains for each join name, an inner dict containing the path to the join dataset
    csv file and its delimiter
    :param init: whether to initialize history
    :param exclude: data features to ignore
    :param n_examples: number of examples to return for each anomaly
    :return: a df with the validation report
    """
    logging.info("Entering runner.validate")
    if join_paths is None:
        join_paths = {}
    if path is None and data is None:
        raise Exception('one of the path or df should be given')

    validation_class = type_class_map[file_type]
    validation_tool = validation_class(file_type, join_paths, n_examples, dq_folder_path)
    logging.info(f"File type: {file_type}")

    if init:
        logging.info("Starting init")
        
        validation_tool.init()

    anomalies = validation_tool.validate_data(data=data, data_path=path, data_delimiter=data_delimiter, exclude=exclude, dq_folder_path = dq_folder_path)
    examples, anomalies = validation_tool.get_anomalies_examples(anomalies=anomalies, data=data, data_path=path,
                                                                 data_delimiter=data_delimiter)

    logging.info("Process finished")
    return examples, anomalies


def add_type_class_matching(file_type: str, custom_validation_class: type):
    """
    add a custom validation class to the type_class_map with the matching file type
    :param file_type: file type
    :param custom_validation_class: custom validation class
    """
    type_class_map[file_type] = custom_validation_class
