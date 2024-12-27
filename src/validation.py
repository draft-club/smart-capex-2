import logging
import os
import pandas as pd
import json
from types import SimpleNamespace
import re
import config.config as config
from datetime import date
import operator
import src.schema_tools as schema_tools
from pandas.api.types import is_string_dtype
import pickle

def update_intent(new_intents, file_type, dq_folder_path= 'dq_report/'):
    """
    This function will update the list of intents and update domain constraint in the schma of the given file_type. If the file intents.pkl does not exists, this function will create it. 
    
    :param new_intents: list of new intents to add to the current intents list
    :param file_type : Name of the schema where the domain constraints needs to be updated
    :param dq_folder_path: Path of the folder where to read and store the intents.pkl file
    """
    if "intents.pkl" in os.listdir(dq_folder_path):
        intents = pickle.load(open(dq_folder_path+"intents.pkl", 'rb'))
        intents = intents + new_intents
        intents = list(set(intents))
    else:
        intents = new_intents
    pickle.dump(intents, open(dq_folder_path+"intents.pkl", 'wb'))
    # Update news intents to schema
    schema = schema_tools.read_schema_from_config(file_type)
    schema = schema_tools.set_domain(schema, 'resp.nlu_intent_name', intents)
    schema_tools.add_schema_to_config(schema, file_type)
        
class Validation:

    def __init__(self, file_type: str, paths=None, n_examples: int = 5, dq_folder_path = None):
        """
        initializer
        :param file_type: could be either oss, cdr, cp, cells or sites
        :param paths: a nested dict with all the needed data to join validations
        :param n_examples: number of anomalous values to return in the anomaly report
        """
        logging.info(f"Initialize Validation object - file_type: {file_type} - n_examples : {n_examples}")
        if paths is None:
            paths = {}
        self.join_paths = paths
        self.file_type = file_type
        if dq_folder_path==None:
            self.schema_path = config.schema_path.format(file_type)
        else:
            self.schema_path = dq_folder_path+'%s_schema.json'%(file_type)
        self.n_examples = n_examples
        self.dq_folder_path = dq_folder_path

    def init(self):
        """
        init history data
        """
        os.remove(config.previous_data.format(self.file_type))

    def load_schema(self) -> object:
        """
        load schema from config file
        :return: json schema in object format
        """
        logging.info(f"Load schema")
        return json.load(open(self.schema_path), object_hook=lambda d: SimpleNamespace(**d))    

    def _get_join_table(self, join_constraint):
        path_detail = self.join_paths[join_constraint.name]
        joined_with_table = pd.read_csv(path_detail['path'],
                                        delimiter=path_detail['delimiter'] if 'delimiter' in path_detail else None)
        return joined_with_table

    def _get_distinctness(self, data: pd.DataFrame, column: str) -> float:
        """
        calculate distinctness for a given column
        :param data: df with data
        :param column: column to calculate distinctness over
        :return: distinctness value
        """
        serie = data.loc[:, column]
        nbr_unique_values = len(serie.unique())
        return 0 if nbr_unique_values == 1 else nbr_unique_values / serie.notna().count()

    def _check_size_drift(self, data: pd.DataFrame, schema: object, previous_data: pd.DataFrame, anomalies: list
                          ) -> list:
        """
        check whether there is a significant difference between the sizes of the two data batches
        :param data: df with data to validate
        :param previous_data: df with data from the previous time span
        :param schema: dict with the constraints for the batch
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if previous_data is None or not hasattr(schema, "dataset_constraint"):
            return anomalies

        if schema.dataset_constraint.max_fraction_threshold or schema.dataset_constraint.max_fraction_threshold:
            actual_fraction = len(data) / len(previous_data)
            if schema.dataset_constraint.max_fraction_threshold:
                max_fraction = schema.dataset_constraint.max_fraction_threshold
                if actual_fraction > max_fraction:
                    msg = f"The ratio of num examples in the current dataset versus the previous span is " \
                          f"{actual_fraction:.6g}, which is above the threshold {max_fraction}"
                    anomalies += [['dataset anomaly', 'Size anomaly', msg, None]]
            elif schema.dataset_constraint.min_fraction_threshold:
                min_fraction = schema.dataset_constraint.min_fraction_threshold
                if actual_fraction < min_fraction:
                    msg = f"The ratio of num examples in the current dataset versus the previous span is " \
                          f"{actual_fraction:.6g}, which is above the threshold {min_fraction}"
                    anomalies += [['dataset anomaly', 'Size anomaly', msg, None]]
        return anomalies

    def _is_numeric(self, x):
        return bool(re.match("^\d+(\.\d+)?$", str(x)))

    def _check_type(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether the feature type is the same as in the schema
        :param data: df with data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if feature.name not in data or not feature.type:
            return anomalies

        def df_types_to_str_types(x):
            return "int" if str(x)[:3] == "int" else "float" if str(x)[:5] == "float" else "str"


        expected_type_to_test = {"int": lambda x: float(x).is_integer() if self._is_numeric(x) else False,
                                 "float": lambda x: self._is_numeric(x),
                                 "str": lambda x: isinstance(x, str)
                                 }
        expected_type = feature.type
        actual_type = df_types_to_str_types(data[feature.name].dtypes)
        anomaly_detected = False
        if expected_type != actual_type:
            msg = f"Expected data of type: {expected_type} but got {actual_type}"
            anomalies += [[feature.name, 'Type anomaly', msg, None]]
            anomaly_detected = True

        if self.n_examples and anomaly_detected:
            examples = []
            for value in data[feature.name]:
                if not expected_type_to_test[expected_type](value):
                    examples += [value]
                    if len(examples) == self.n_examples:
                        break
            anomalies[-1][-1] = examples
        return anomalies

    def _check_presence(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether the feature satisfy the presence requirements
        :param data: df with data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if feature.name not in data:
            actual_presence = 0
        elif feature.presence > 1:
            actual_presence = len(data) - data[feature.name].isna().sum()
        else:
            actual_presence = 1 - data[feature.name].isna().sum() / len(data)
        if actual_presence < feature.presence:
            msg = "The feature was present in fewer examples than expected: " \
                  "minimum = {:.6g}, actual = {:.6g}"
            anomalies += [[feature.name, 'Presence anomaly', msg.format(feature.presence, actual_presence), None]]
        return anomalies

    def _check_domain(self, data: pd.DataFrame, feature: object, anomalies: list, dq_folder_path = 'dq_report/') -> list:
        """
        check whether the feature respect the domain constraints defined in the schema
        :param data: df with data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if feature.name not in data:
            return anomalies

        if isinstance(feature.domain, list):
            unique_values = data[feature.name].unique()
            anomalous_values = [x for x in unique_values if str(x) not in [str(y) for y in feature.domain]]
            if anomalous_values:
                msg = f"Examples contain values missing from the schema: {str(anomalous_values[:5])[1:-1]}"
                anomalies += [[feature.name, 'Unexpected string values', msg, anomalous_values[:self.n_examples]]]
            #Export anomalous values to pickle
            #pickle.dump(anomalous_values, open('%sanomalous_values.pkl'%(dq_folder_path),'wb'))
            
            # Update intents.pkl file and schema
            #update_intent(new_intents = anomalous_values, file_type = self.file_type, dq_folder_path = dq_folder_path)
       
        elif isinstance(feature.domain, SimpleNamespace):
            if is_string_dtype(data[feature.name]):
                return anomalies
            sorted_feature = data[feature.name].sort_values()
            max_value = feature.domain.max if hasattr(feature.domain, 'max') and\
                                              feature.domain.max != '' else float("inf")

            min_value = feature.domain.min if hasattr(feature.domain, 'min') and\
                                              feature.domain.min != '' else float("-inf")

            actual_max = sorted_feature.values[-1]
            actual_min = sorted_feature.values[0]
            if max_value < actual_max:
                examples = [x for x in sorted_feature[-self.n_examples:] if max_value < x]
                msg = f"Unexpectedly big value: {actual_max}"
                anomalies += [[feature.name, 'Out-of-range values', msg, examples]]
            if min_value > actual_min:
                examples = [x for x in sorted_feature[:self.n_examples] if min_value > x]
                msg = f"Unexpectedly small value: {actual_min}"
                anomalies += [[feature.name, 'Out-of-range values', msg, examples]]
        return anomalies

    def _check_drift(self, data: pd.DataFrame, feature: object, previous_data: pd.DataFrame, anomalies: list, dq_folder_path = 'dq_report/') -> list:
        """
        check whether there is a significant drift between two data batches
        :param data: df with data to validate
        :param previous_data: df with data from the previous time span
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if previous_data is None or feature.name not in data or feature.name not in previous_data\
                or feature.drift in [None, ""]:
            return anomalies

        actual_histogram = data.groupby(feature.name).size()
        previous_histogram = previous_data.groupby(feature.name).size()
        actual_frequency_distribution = actual_histogram / actual_histogram.sum()
        previous_frequency_distribution = previous_histogram / previous_histogram.sum()
        for key in actual_frequency_distribution.keys():
            if key not in previous_frequency_distribution:
                previous_frequency_distribution[key] = 0
        for key in previous_frequency_distribution.keys():
            if key not in actual_frequency_distribution:
                actual_frequency_distribution[key] = 0
        histogram_subtraction = abs(actual_frequency_distribution - previous_frequency_distribution)
        sorted_histogram_subtraction = histogram_subtraction.nlargest()
        pickle.dump(sorted_histogram_subtraction, open('%sdrift.pkl'%(dq_folder_path), 'wb'))
        drift = sorted_histogram_subtraction.iloc[0]
        if drift > feature.drift:
            msg = f"The Linfty distance between the two batches is {drift:.6g}, " \
                  f"above the threshold {feature.drift}. " \
                  f"The feature value with maximum difference is: {sorted_histogram_subtraction.keys()[0]}" \
                  f" The Top 5 drifts values are : "
            top_5_drifts = {}
            for i in range(5):
                top_5_drifts[sorted_histogram_subtraction.keys()[i]] = round(sorted_histogram_subtraction.iloc[i],5)
            msg = msg + str(top_5_drifts)
            anomalies += [[feature.name, 'Drift anomaly', msg, sorted_histogram_subtraction]]
        return anomalies

    def _check_pattern(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether there is any values that does not match their pattern
        :param data: data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        examples = []
        if feature.regex:
            schema_regex = feature.regex.replace('\\\\', '\\')
            for i in range(len(data)):
                regex = schema_regex
                row = data.iloc[i, :]
                value = row[feature.name]
                placeholders = re.findall('#[^#]*#', regex)
                for placeholder in placeholders:
                    n = placeholder.find('[')
                    if n == -1:
                        if placeholder[1:-1] not in data:
                            regex = regex.replace(placeholder, "")
                        else:
                            regex = regex.replace(placeholder, re.escape(str(row[placeholder[1:-1]])))
                    else:
                        if placeholder[1:n] not in data:
                            regex = regex.replace(placeholder, "")
                        else:
                            indexes = re.findall('[0-9]+', placeholder[n+1:])
                            indexes = list(map(int, indexes))
                            val = re.escape(str(row[placeholder[1:n]]))

                            regex = regex.replace(placeholder,
                                                  val[indexes[0]: indexes[1]+1])
                result = re.match(regex, str(value).strip())
                if result is None:
                    examples += [value]
                    if self.n_examples and len(examples) == self.n_examples:
                        break
        if examples:
            anomalies += [[feature.name, 'Pattern anomaly',
                           'At least one value: {0} does NOT match pattern : {1}'.format(examples[0],
                                                                                         schema_regex), examples]]
        return anomalies

    def _check_distinctness(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether the distinctness of the feature respect a certain condition (equal to, above or below a threshold)
        :param data: data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        str_to_op = {
            'lt': (operator.lt, 'is above'),
            'eq': (operator.eq, 'is not equal'),
            'gt': (operator.gt, 'is below')
        }
        if feature.distinctness:
            condition = feature.distinctness.condition
            threshold = feature.distinctness.value
            distinctness = self._get_distinctness(data, feature.name)
            if not str_to_op[condition][0](distinctness, threshold):
                anomalies += [[feature.name, 'Distinctness anomaly',
                               'Actual distinctness value: {} {} '.format(distinctness, str_to_op[condition][1]) +
                               'threshold: {}'.format(threshold), None]]
                if condition == 'eq':
                    if threshold == 1:
                        duplicates = data[feature.name][data[feature.name].duplicated()]
                        anomalies[-1][-1] = [x for x in duplicates[:self.n_examples]]
                    if threshold == 0:
                        anomalies[-1][-1] = [x for x in data[feature.name].unique()[:self.n_examples]]
        return anomalies

    def _check_outliers(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether there is any outliers in the data
        :param data: data to validate
        :param anomalies: list with current anomaly report
        :return: a list with to appended custom anomalies report
        """
        if feature.outliers:
            mean = data[feature.name].mean()
            std = data[feature.name].std()
            cmax = data[feature.name].max()
            cmin = data[feature.name].min()
            outliers = feature.outliers
            if cmin < mean - outliers * std:
                anomalies += [[feature.name, 'Outlier anomaly', 'Value {} is considered as an anomaly'.format(cmin),
                               [cmin]]]

            if cmax < mean + outliers * std:
                anomalies += [[feature.name, 'Outlier anomaly', 'Value {} is considered as an anomaly'.format(cmax),
                               [cmax]]]
        return anomalies

    def _check_highest_frequency_value(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check whether the highest frequent value within a feature has a frequency that does not over a threshold
        :param data: df with data _to _validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        threshold = feature.highest_frequency_threshold
        if threshold and feature.type != "str":
            message = 'Highest frequency should not be defined for numerical feature {}'.format(feature.name)
            anomalies += [['schema anomaly', 'Schema anomaly', message]]
            return anomalies

        if threshold and threshold < 1:
            histogram = data.groupby(feature.name).size()
            frequency = histogram / histogram.sum()
            highest_frequency = frequency.nlargest()
            ratio = highest_frequency.values[0]
            value = highest_frequency.keys()[0]
            if ratio > threshold:
                examples = [key for key, value in highest_frequency[:self.n_examples].items() if value > threshold]
                anomalies += [[feature.name, 'Too frequent value anomaly',
                               'The value "{}" represent '
                               'a ratio of {} of non missing data which '.format(value, ratio) +
                               'is over threshold: {}'.format(threshold), examples]]
        elif threshold and threshold >= 1:
            histogram = data.groupby(feature.name).size()
            highest_frequency = histogram.nlargest()
            number = highest_frequency.values[0]
            value = highest_frequency.keys()[0]
            if number > threshold:
                examples = [key for key, value in highest_frequency[:self.n_examples].items() if value > threshold]
                anomalies += [[feature.name, 'Too frequent value anomaly',
                               'The frequency of the value "{}" '.format(value) +
                               'is {} which is over '.format(number) +
                               'threshold: {}'.format(threshold), examples]]
        return anomalies

    def _check_mapping(self, data: pd.DataFrame, feature: object, anomalies: list) -> list:
        """
        check if a feature values are mapped only to one value of another feature
        :param data: data to validate
        :param feature: dict with the constraints for the feature
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        mapped_to = feature.mapped_to_only_one
        if type(mapped_to) is str:
            if mapped_to != "":
                if mapped_to not in data.columns:
                    message = 'The value of \'mapped_to_only_one\' feature is not in data'
                    anomalies += [['schema anomaly', 'Schema anomaly', message]]
                else:
                    mapping_sorted_count = data.groupby(feature.name)[mapped_to].nunique().sort_values(ascending=False)

                    if not mapping_sorted_count.empty and mapping_sorted_count.iloc[0] > 1:
                        examples = [key for key, value in mapping_sorted_count[:self.n_examples].items() if value > 1]
                        feature_value = mapping_sorted_count.keys()[0]
                        message = 'At least one value ({}) is mapped to multiple {} values'.format(feature_value,
                                                                                                   mapped_to)
                        anomalies += [[feature.name, 'Mapping anomaly', message, examples]]

        elif type(mapped_to) is list:
            for to_feature in mapped_to:
                if to_feature not in data.columns:
                    message = 'One of the \'mapped to one value of\' features are not in data'
                    anomalies += [['schema anomaly', 'Schema anomaly', message]]
                    continue
                mapping_sorted_count = data.groupby(feature.name)[to_feature].nunique().sort_values(ascending=False)
                if not mapping_sorted_count.empty and mapping_sorted_count.iloc[0] > 1:
                    examples = [key for key, value in mapping_sorted_count[:self.n_examples].items() if value > 1]
                    feature_value = mapping_sorted_count.keys()[0]
                    message = 'At least one value ({}) is mapped to multiple {} values'.format(feature_value,
                                                                                               to_feature)
                    anomalies += [[feature.name, 'Mapping anomaly', message, examples]]
        else:
            message = 'value of \'mapped to one value of\' is neither list of strings nor a string'
            anomalies += [[feature.name, 'Schema anomaly', message, None]]
        return anomalies

    def _check_new_features(self, data: pd.DataFrame, schema: object, anomalies: list) -> list:
        """
        check if data contains features that are missing in the schema 
        :param data: data to validate
        :param schema: dict with the constraints for the dataset
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        schema_feats = [feature.name for feature in schema.features_constraint]
        new_feats = [feat for feat in data if feat not in schema_feats]
        if len(new_feats) > 0:
            for feat in new_feats:
                anomalies += [[feat, 'Schema anomaly', 'The feature is in data but missing from the schema', None]]
        return anomalies

    def _check_unicity(self, data: pd.DataFrame, schema: object, anomalies: list) -> list:
        """
        check whether there are any duplicates in the data
        :param data: data to validate
        :param schema: dict with the constraints for the dataset
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if not hasattr(schema, 'dataset_constraint') or not hasattr(schema.dataset_constraint, 'unicity_features'):
            return anomalies
        unicity_features = schema.dataset_constraint.unicity_features
        if not unicity_features:
            return anomalies
        if type(unicity_features) is str and unicity_features == "*":
            duplicates = data[data.duplicated()]
        elif type(unicity_features) is str and unicity_features in data:
            duplicates = data[data.duplicated([unicity_features])]
        elif type(unicity_features) is str and unicity_features not in data:
            message = 'Features given to check duplicates are not in present in the data to validate'
            anomalies += [['schema anomaly', 'Schema anomaly', message, None]]
            return anomalies
        elif type(unicity_features) is list:
            try:
                duplicates = data[data.duplicated(unicity_features)]
            except KeyError:
                message = 'Features given to check duplicates are not in present in the data to validate'
                anomalies += [['schema anomaly', 'Schema anomaly', message, None]]
                return anomalies
        else:
            message = 'schema.dataset_constraint.unicity_features field in the json file should be "*" or a list of ' \
                      'features'
            anomalies += [['schema anomaly', 'Schema anomaly', message, None]]
            return anomalies
        if len(duplicates) > 0:
            condition = [not value for value in duplicates.duplicated(unicity_features)]
            one_copy_each = duplicates[condition][unicity_features]
            examples = pd.DataFrame(one_copy_each).to_dict("records")[:self.n_examples]
            anomaly_massage = 'There is duplicates in the dataset, the features used to check are {}'
            if unicity_features == "*":
                anomaly_massage = anomaly_massage.format('all the dataset features')
            else:
                anomaly_massage = anomaly_massage.format(unicity_features)
            anomalies += [['dataset anomaly', 'Duplicates anomaly', anomaly_massage, examples]]
        return anomalies

    def _check_join_quality(self, data: pd.DataFrame, schema: object, anomalies: list) -> list:
        """
        check if the ratio of the size of the dataset that result from the join between this one and another one
        compared to the size of the actual dataset is above a threshold
        :param data: data to validate
        :param schema: dict with the constraints for the dataset
        :param anomalies: list with current anomaly report
        :return: a list with appended anomalies report
        """
        if not hasattr(schema, 'dataset_constraint') or not hasattr(schema.dataset_constraint, 'joins_constraint') or \
                not schema.dataset_constraint.joins_constraint:
            return anomalies

        for join_constraint in schema.dataset_constraint.joins_constraint:
            if join_constraint.name not in self.join_paths:
                anomaly_massage = '{} was not given in the join_paths arg'.format(join_constraint.name)
                anomalies += [['schema anomaly', 'Schema anomaly', anomaly_massage, None]]
                continue
            joined_with_table = self._get_join_table(join_constraint)
            if join_constraint.left_on not in data or join_constraint.right_on not in joined_with_table:
                anomaly_massage = 'At least one of the join features defined in the schema for join {} is not found'
                anomaly_massage = anomaly_massage.format(join_constraint.name)
                anomalies += [['dataset anomaly', 'Schema anomaly', anomaly_massage, None]]
                continue
            keys = data[join_constraint.left_on].unique()
            joined_with_keys = joined_with_table[join_constraint.right_on].unique()
            join_quality = len([x for x in keys if x in joined_with_keys]) / len(keys)
            if join_quality < join_constraint.threshold:
                anomaly_massage = 'The quality of the join between the dataset and the "{}" dataset is {}, which is ' \
                                  'below the threshold {}'
                anomaly_massage = anomaly_massage.format(join_constraint.name, join_quality, join_constraint.threshold)
                anomalies += [['dataset anomaly', 'Join anomaly', anomaly_massage, None]]
        return anomalies

    def _validate_basic_constraints(self, data: pd.DataFrame, schema: object, previous_data: pd.DataFrame,
                                    nested: bool = False, dq_folder_path = 'dq_report/') -> pd.DataFrame:
        """
        validate general constraint
        :param data: data to validate
        :param schema: json schema with the constraints
        :param previous_data: last span data
        :param nested: whether the validation is for a slice
        :return: a df with an appended anomaly report
        """
        logging.info(f"validate basic constraints")
        anomalies = []
        logging.info(f"Check unicity")
        anomalies = self._check_unicity(data, schema, anomalies)
        logging.info(f"Check join quality")
        anomalies = self._check_join_quality(data, schema, anomalies)
        logging.info(f"Check dataset size drift")
        anomalies = self._check_size_drift(data, schema, previous_data, anomalies)
        logging.info(f"Check new features")
        anomalies = self._check_new_features(data, schema, anomalies) if not nested else anomalies
        for feature in schema.features_constraint:
            logging.info(f"Feature {feature.name} : Check presence")
            anomalies = self._check_presence(data, feature, anomalies)

            # ignore dropped:
            if feature.name not in data.columns:
                continue

            logging.info(f"Feature {feature.name} : Check type")
            anomalies = self._check_type(data, feature, anomalies)

            logging.info(f"Feature {feature.name} : Check domain")
            anomalies = self._check_domain(data = data, feature = feature, anomalies = anomalies, dq_folder_path = dq_folder_path)

            logging.info(f"Feature {feature.name} : Check drift")
            anomalies = self._check_drift(data = data, feature = feature, previous_data = previous_data, anomalies = anomalies, dq_folder_path = dq_folder_path)

            logging.info(f"Feature {feature.name} : Check distinctness")
            anomalies = self._check_distinctness(data, feature, anomalies)

            logging.info(f"Feature {feature.name} : Check pattern")
            anomalies = self._check_pattern(data, feature, anomalies)

            logging.info(f"Feature {feature.name} : Check outliers")
            anomalies = self._check_outliers(data, feature, anomalies)

            logging.info(f"Feature {feature.name} : Check highest frequency value")
            anomalies = self._check_highest_frequency_value(data, feature, anomalies)

            logging.info(f"Feature {feature.name} : Check mapping")
            anomalies = self._check_mapping(data, feature, anomalies)

        anomalies_df_columns = ["feature_name", "anomaly_short_description", "anomaly_long_description", "examples"]
        anomalies_df = pd.DataFrame(anomalies, columns=anomalies_df_columns)
        return anomalies_df

    def _validate_slices(self, data: pd.DataFrame, schema: object, previous_data: pd.DataFrame,
                         anomalies_df: pd.DataFrame, dq_folder_path = 'dq_report/') -> pd.DataFrame:
        """
        validate constraints for a slice of data
        :param data: data to validate
        :param schema: json schema with the constraints
        :param previous_data: data from the last timespan
        :param anomalies_df: df with basic validation anomaly report
        :return: a df with a global anomaly report
        """
        if hasattr(schema, 'slices_constraint') and schema.slices_constraint:
            for constraint in schema.slices_constraint:
                slicing_column = constraint.slicing_column
                values_to_take = constraint.values_to_take
                values_to_drop = constraint.values_to_drop
                if slicing_column not in data:
                    continue
                data_slice = data.query('{} in {}'.format(slicing_column, values_to_take)) if values_to_take else data
                data_slice = data_slice.query('{} not in {}'.format(slicing_column,
                                                                    values_to_drop)) if values_to_drop else data_slice

                previous_data_slice = None
                if previous_data is not None and slicing_column in previous_data:
                    previous_data_slice = previous_data.query('{} in {}'.format(slicing_column, values_to_take))\
                        if values_to_take else previous_data
                    previous_data_slice = previous_data_slice.query('{} not in {}'.format(slicing_column,
                                                                                          values_to_drop)) \
                        if values_to_drop else previous_data_slice

                inner_schema = constraint.schema
                logging.info(f"Validate slice - Slicing column : {slicing_column} - values to take: {values_to_take} "
                             f"- values to drop: {values_to_drop}")
                anomalies = self._validate_basic_constraints(data_slice, inner_schema, previous_data_slice, True, dq_folder_path = dq_folder_path )
                anomalies = self._validate_custom_constraints(data_slice, inner_schema, anomalies)
                anomalies = self._validate_slices(data_slice, inner_schema, previous_data, anomalies, dq_folder_path = dq_folder_path)

                anomalies["anomaly_long_description"] = 'Sliced by {}: '.format(slicing_column) + \
                                                        anomalies["anomaly_long_description"]

                anomalies['examples'] = [{'feature': slicing_column, 'values_to_take': values_to_take,
                                         'values_to_drop': values_to_drop, 'examples': example}
                                         for example in anomalies.examples]

                anomalies_df = pd.concat([anomalies_df, anomalies])

        return anomalies_df

    def _validate_custom_constraints(self, data: pd.DataFrame, schema: object,
                                     anomalies_df: pd.DataFrame) -> pd.DataFrame:
        return anomalies_df

    def validate_data(self, data: pd.DataFrame = None, data_path: str = None, previous_data: pd.DataFrame = None,
                      previous_data_path: str = None, data_delimiter: str = ',', previous_data_delimiter: str = ',',
                      exclude: list = None, dq_folder_path = None) -> pd.DataFrame:
        """
        ###### Module main function:
        function to validate data
        :param data: data as a dataframe
        :param data_path: path to data
        :param previous_data: previous data as a dataframe
        :param previous_data_path: path to previous data
        :param previous_data_delimiter: delimiter for data file
        :param data_delimiter: delimiter for previous data file
        :param exclude: list of data feature not to validate
        :return: an anomaly report
        """
        logging.info(f"Start validation ")
        if data is None and not data_path:
            raise Exception('data and data_path are null')

        if data is None and data_path:
            data = pd.read_csv(data_path, delimiter=data_delimiter)

        if previous_data is None and previous_data_path:
            previous_data = pd.read_csv(previous_data_path, delimiter=previous_data_delimiter)
        elif previous_data is None and previous_data_path in [None, ""]:
            try:
                if dq_folder_path == None: 
                    previous_data = pd.read_csv(config.previous_data.format(self.file_type))
                else:
                     previous_data = pd.read_csv(dq_folder_path+'previous_%s.csv'%(self.file_type))   
            except Exception:
                previous_data = None

        schema = self.load_schema()

        if exclude:
            for feature in exclude:
                data = data.drop(feature, axis=1)
                previous_data = previous_data.drop(feature, axis=1) if previous_data is not None else None
                schema_tools.remove_feature_constraint(schema, feature)

        anomalies = self._validate_basic_constraints(data, schema, previous_data, dq_folder_path = 'dq_report/')
        anomalies = self._validate_slices(data, schema, previous_data, anomalies, dq_folder_path = 'dq_report/')
        anomalies = self._validate_custom_constraints(data, schema, anomalies)

        anomalies['date'] = [date.today()] * len(anomalies)
        if schema.dataset_constraint.batch_id_column:
            anomalies['batch_id'] = [data[schema.dataset_constraint.batch_id_column].iloc[0]] * len(anomalies)
        else:
            anomalies['batch_id'] = anomalies['date'].map(str)

        anomalies['type'] = [self.file_type] * len(anomalies)
        if dq_folder_path == None:
            data.to_csv(config.previous_data.format(self.file_type))
        else:
            data.to_csv(dq_folder_path+'previous_%s.csv'%(self.file_type))
        return anomalies

    def get_anomalies_examples(self, anomalies: pd.DataFrame, data: pd.DataFrame = None, data_path: str = "",
                               data_delimiter=",") -> (pd.DataFrame, pd.DataFrame):
        """
        get a df with examples of each anomaly detected
        :param anomalies: df with the anomalies
        :param data: df with data that gave the anomaly report
        :param data_path: path to data that gave the anomaly report
        :param data_delimiter: delimiter of the data file
        :return: a df with the examples and an anomaly df with an index that allows the matching anomaly-example
        """
        logging.info(f"Get anomalies examples")
        if data is None and not data_path:
            raise Exception('data and data_path are null')

        if data is None and data_path:
            data = pd.read_csv(data_path, delimiter=data_delimiter)
        if anomalies.empty:
            anomalies['id'] = ""
        else:
            anomalies = anomalies.sort_values(by=['feature_name', 'anomaly_short_description'])\
                .reset_index(drop=True).copy()
            anomalies['id'] = anomalies['type'] + '-' + anomalies['batch_id'].map(str) + '-' + \
                              anomalies['date'].map(str) + '-' + [str(x) for x in range(len(anomalies))]

        anomalies = anomalies.set_index('id')
        return_df = pd.DataFrame(columns=['anomaly_id', 'dv_example'])
        for index, anomaly in anomalies.iterrows():
            feature = anomaly.feature_name
            examples = anomaly.examples
            short_desc = anomaly.anomaly_short_description
            if feature not in data and feature != 'dataset anomaly':
                continue
            filtered = self._get_anomaly_example(examples, data, feature, short_desc)
            if filtered is None:
                continue
            filtered = filtered.sample(self.n_examples if len(filtered) > self.n_examples else len(filtered))
            filtered['anomaly_id'] = [index] * len(filtered)
            return_df = pd.concat([return_df, filtered])
        if not return_df.empty:
            columns_to_convert = list(return_df.columns)
            columns_to_convert.remove("anomaly_id")
            columns_to_convert.remove("dv_example")
            return_df['dv_example'] = list(map(str, return_df[columns_to_convert].to_dict(orient='records')))
            return_df = return_df[['anomaly_id', 'dv_example']]
        logging.info(f"Return final anomaly report + examples")
        return return_df, anomalies

    def _get_anomaly_example(self, examples: object, data: pd.DataFrame, feature: str, short_desc: str) -> pd.DataFrame:
        """
        get rows from the data df with the "examples" values
        :param examples: example values of the anomaly
        :param data: df that gave the anomaly report
        :param feature: feature ith the anomaly
        :param short_desc: the anomaly short description
        :return: a df with the selected rows
        """
        if not examples and short_desc != "Presence anomaly":
            return None
        elif short_desc == "Presence anomaly":
            filtered = data[data[feature].isna()]
        elif isinstance(examples, list) and not isinstance(examples[0], dict):
            filtered = data[data[feature].isin(examples)]
        elif isinstance(examples, list) and isinstance(examples[0], dict):
            filtered = pd.DataFrame()
            for example in examples:
                filtered_ex = data.loc[(data[list(example)] == pd.Series(example)).all(axis=1)]
                filtered = pd.concat([filtered, filtered_ex])
        else:
            slicing_feature = examples.feature
            values_to_take = examples.values_to_take
            values_to_drop = examples.values_to_drop
            examples = examples.examples
            data_slice = data.query('{} in {}'.format(slicing_feature, values_to_take)) if values_to_take else data
            data_slice = data_slice.query('{} not in {}'.format(slicing_feature,
                                                                values_to_drop)) if values_to_drop else data_slice
            filtered = self._get_anomaly_example(examples, data_slice, feature, short_desc)
        return filtered
