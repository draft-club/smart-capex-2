import logging
import pandas as pd
import os
import json
from src.validation import Validation
import config.config as config
import numpy as np

class OssValidation(Validation):

    def __init__(self, file_type, paths=None, n_examples: int = 5):
        """
        initializer
        :param file_type: could be either oss, cdr, cp, cells or sites
        :param paths: a nested dict with all the needed data to join validations
        :param n_examples: number of anomalous values to return in the anomaly report
        """
        if paths is None:
            paths = {}
        super().__init__(file_type, paths, n_examples)
        self.schema = self.load_schema()
        self.time_series_history = None
        try:
            with open(config.time_series_history.format(file_type), 'r') as f:
                self.time_series_history = json.load(f)
        except FileNotFoundError:
            self.time_series_history = {}

    def init(self):
        super().init()
        # init time series history
        with open(config.time_series_history.format(self.file_type), 'w') as f:
            json.dump({}, f)

        # init previous weeks cells list:
        with open(config.cells_list.format(self.file_type), "w") as cells_file:
            cells_file.write("")

    def _check_number_of_new_removed_cells(self, data: pd.DataFrame, schema: object,
                                           custom_anomalies: list) -> list:
        """
        check whether the number of new cells / disappeared cells from a week to another is not above a threshold
        :param data: data to validate
        :param schema: json schema with the constraints
        :param custom_anomalies: df with current validation anomaly report
        :return: a df with to appended custom anomalies report
        """
        cell_name_feature = self.schema.custom_constraint.cell_name_feature
        if hasattr(schema, 'custom_constraint') and \
                (hasattr(schema.custom_constraint, 'max_new_cells') or
                 hasattr(schema.custom_constraint, 'max_disappeared_cells')) and\
                (schema.custom_constraint.max_new_cells != "" or
                 schema.custom_constraint.max_disappeared_cells != "") and \
                cell_name_feature in data:

            current_batch_cells = data[cell_name_feature].unique()
            if os.path.isfile(config.cells_list.format(self.file_type)):
                with open(config.cells_list.format(self.file_type), "r") as f:
                    old_cells = f.read().split("\n")

                if hasattr(schema.custom_constraint, 'max_new_cells') and\
                    isinstance(schema.custom_constraint.max_new_cells, (int, float)):
                    new_cells = set(current_batch_cells) - set(old_cells)
                    nbr_new_cells = len(new_cells)
                    if nbr_new_cells > schema.custom_constraint.max_new_cells:
                        limit = schema.custom_constraint.max_new_cells
                        custom_anomalies += [[cell_name_feature, 'To many new cells',
                                              'There is {} new cell compared to last batch '.format(nbr_new_cells) +
                                              'which is greater than the limit: {}'.format(limit),
                                              list(new_cells)[:self.n_examples]]]

                if hasattr(schema.custom_constraint, 'max_disappeared_cells') and \
                        isinstance(schema.custom_constraint.max_disappeared_cells, (int, float)):

                    disappeared_cells = set(old_cells) - set(current_batch_cells)
                    nbr_disappeared_cells = len(disappeared_cells)
                    if nbr_disappeared_cells > schema.custom_constraint.max_disappeared_cells:
                        limit = schema.custom_constraint.max_disappeared_cells
                        custom_anomalies += [[cell_name_feature, 'To many disappearing cells',
                                              'There is {} disappeared cell '.format(nbr_disappeared_cells) +
                                              'compared to last batch ' +
                                              'which is greater than the limit: {}'.format(limit),
                                              list(disappeared_cells)[:self.n_examples]]]

            # save cells list for upcoming batches
            with open(config.cells_list.format(self.file_type), 'w') as cells_file:
                cells_file.write("\n".join(current_batch_cells))
        return custom_anomalies

    def _check_number_of_cell_per_site(self, data: pd.DataFrame, schema: object,
                                       custom_anomalies: list) -> list:
        """
        check whether the number of cells per site is not above a threshold
        :param data: data to validate
        :param schema: json schema with the constraints
        :param custom_anomalies: df with current anomaly report
        :return: a df with to appended custom anomalies report
        """
        if hasattr(schema, 'custom_constraint') and hasattr(schema.custom_constraint, 'max_cell_per_site'):
            if schema.custom_constraint.max_cell_per_site and 'site_id' in data:
                cell_per_site = data.groupby("site_id").size().sort_values(ascending=False)
                limit = schema.custom_constraint.max_cell_per_site
                if cell_per_site[0] > limit:
                    examples = [key for key, value in cell_per_site[:self.n_examples].items() if value > limit]
                    custom_anomalies += [['site_id', 'Cell per site anomaly',
                                          'There is {} cell for the site: {} '.format(cell_per_site[0],
                                                                                      cell_per_site.keys()[0]) +
                                          'which is greater than the limit: {}'.format(limit), examples]]
        return custom_anomalies

    def _check_outliers(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether there is any outliers in the feature. The outliers are detected compared to the old values of the
        same cell_name
        :param data: data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: df with current anomaly report
        :return: a df with to appended custom anomalies report
        """
        cell_name_feature = self.schema.custom_constraint.cell_name_feature
        if feature.outliers and cell_name_feature in data:
            if feature.type == "str":
                anomalies += [[feature.name, 'Schema anomaly', 'Cannot detect outliers for String values']]
            else:
                if feature.name not in self.time_series_history:
                    self.time_series_history[feature.name] = {}
                feature_history = self.time_series_history[feature.name]

                for i in range(len(data)):
                    cell_name = data.iloc[i, :][cell_name_feature]
                    value = data.iloc[i, :][feature.name]
                    if value != value or value is None or not self._is_numeric(value):
                        continue
                    value = float(value)
                    if cell_name in feature_history:
                        measure = feature_history[cell_name]

                        if measure['nbr'] > 10 and (
                                value < measure['mean'] - feature.outliers * measure['mean'] or
                                value > measure['mean'] + feature.outliers * measure['mean']):
                            examples = [{"cell_name": cell_name, feature.name: value}]
                            msg = 'Value {} for cell {} is considered as an anomaly'.format(value, cell_name)
                            anomalies += [[feature.name, 'Outlier anomaly', msg, examples]]

                        # update measures
                        measure['mean'] = (measure['mean'] * measure['nbr'] + value) / (measure['nbr'] + 1)
                        measure['squared_sum'] = measure['squared_sum'] + value ** 2
                        measure['nbr'] += 1
                        variance = (measure['squared_sum'] / measure['nbr'] - measure['mean'] ** 2)
                        measure['std'] = variance ** 0.5 if not np.isclose(variance, 0, atol=1e-04) else 0


                    else:
                        init = {'mean': value, 'squared_sum': value ** 2, 'nbr': 1, 'std': 0}
                        self.time_series_history[feature.name][cell_name] = init
        return anomalies

    def _validate_custom_constraints(self, data: pd.DataFrame, schema: object,
                                     anomalies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function that do all the non-generic validations that we developed, and return an anomaly report
        :param data: data to validate
        :param schema: json object with all the constraints
        :param anomalies_df: df with current anomaly report
        :return: a df with appended custom anomalies report
        """
        custom_anomalies = []
        logging.info(f"Check number of new and removed cells")
        custom_anomalies = self._check_number_of_new_removed_cells(data, schema, custom_anomalies)
        logging.info(f"Check number of cell per site")
        custom_anomalies = self._check_number_of_cell_per_site(data, schema, custom_anomalies)
        custom_anomalies_df_columns = ["feature_name", "anomaly_short_description", "anomaly_long_description",
                                       "examples"]
        custom_anomalies_df = pd.DataFrame(custom_anomalies, columns=custom_anomalies_df_columns)
        anomalies_df = pd.concat([anomalies_df, custom_anomalies_df])

        # save
        try:
            with open(config.time_series_history.format(self.file_type), 'w') as f:
                json.dump(self.time_series_history, f, indent=4)
        except FileNotFoundError:
            os.mkdir(config.history_folder)
            with open(config.time_series_history.format(self.file_type), 'w') as f:
                json.dump(self.time_series_history, f, indent=4)

        return anomalies_df
