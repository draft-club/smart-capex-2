import os
from pathlib import Path
from typing import Union
import pandas as pd
import json
from types import SimpleNamespace
import re


def load_schema(path) -> object:
    """
    load schema from config file
    :return: json schema in object format
    """
    return json.load(open(path), object_hook=lambda d: SimpleNamespace(**d))


def _get_dict(element):
    """
    convert a schema / schema element from SimpleNameSpace to dict
    :param element: schema / schema element to convert
    :return: a matching dict / list of dicts for the given element
    """
    if not isinstance(element, (SimpleNamespace, list)):
        return element
    result_dict = None
    if isinstance(element, SimpleNamespace):
        result_dict = vars(element)
        for x in result_dict:
            if isinstance(result_dict[x], (SimpleNamespace, list)):
                result_dict[x] = _get_dict(result_dict[x])
    if isinstance(element, list):
        result_dict = []
        for x in element:
            result_dict += [_get_dict(x)]
    return result_dict


def save_schema(schema: SimpleNamespace, path: str):
    """
    save the schema into the given path
    :param schema: schema to save
    :param path: path to save in
    """
    with open(path, 'w') as f:
        json.dump(_get_dict(schema), f, indent=4)

def add_schema_to_path(schema: SimpleNamespace, file_type: str, path = 'dq_report/'):
    """
    save the schema into config
    :param schema: schema to save
    :param file_type: file for the schema
    """
    if schema:
        with open(path + '%s_schema.json'%(file_type), 'w') as f:
            json.dump(_get_dict(schema), f, indent=4)

def read_schema_from_config(file_type: str):
    """
    save the schema into config
    :param schema: schema to save
    :param file_type: file for the schema
    """
    current_path = str(os.path.abspath(__file__))
    path = Path(current_path)
    root_dir = str(path.parent.parent.absolute())
    with open(f'{root_dir}/config/{file_type}_schema.json', 'rb') as f:
        schema = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        return schema
            
def add_schema_to_config(schema: SimpleNamespace, file_type: str):
    """
    save the schema into config
    :param schema: schema to save
    :param file_type: file for the schema
    """
    if schema:
        current_path = str(os.path.abspath(__file__))
        path = Path(current_path)
        root_dir = str(path.parent.parent.absolute())
        with open(f'{root_dir}/config/{file_type}_schema.json', 'w') as f:
            json.dump(_get_dict(schema), f, indent=4)


def _get_feature_type(data: pd.DataFrame, feature: str) -> str:
    """
    function that return the type of the feature used in the schema
    :param data: dataframe with the data
    :param feature: the concerned feature
    :return: the type of the feature
    """
    dtype = data.dtypes[feature]
    if 'float' in str(dtype):
        return "float"
    if 'int' in str(dtype):
        return "int"
    return "str"


def _get_constraints_list_schema(schema: SimpleNamespace, body: list = None, pref: str = ""):
    """
    a function that cast the constraints in the schema to a list with only the applied validations
    :param schema: schema with all the constraints
    :param body: list that would nbe appended to contain the validations on the schema
    :param pref: the inner schema's prefix
    """
    if body is None:
        body = []
    if hasattr(schema, 'dataset_constraint') and schema.dataset_constraint:
        if schema.dataset_constraint.min_fraction_threshold:
            message = 'the fraction of the size of the current batch compared with the previous one should be'
            body += [['dataset', pref + message + ' above {}'.format(schema.dataset_constraint.min_fraction_threshold)]]
        if schema.dataset_constraint.max_fraction_threshold:
            message = 'the fraction of the size of the current batch compared with the previous one should be'
            body += [['dataset', pref + message + ' below {}'.format(schema.dataset_constraint.max_fraction_threshold)]]
        if schema.dataset_constraint.unicity_features:
            if schema.dataset_constraint.unicity_features == "*":
                unicity_features = ""
            else:
                unicity_features = "'s subset with only the features {}".format(
                    schema.dataset_constraint.unicity_features)
            body += [['dataset', pref + 'dataset{} should not contain duplicate rows'.format(unicity_features)]]
        if schema.dataset_constraint.joins_constraint:
            for join_constraint in schema.dataset_constraint.joins_constraint:
                body += [['dataset', f'{pref}join constraint "{join_constraint.name}" '
                                     f'on the {join_constraint.left_on} feature']]
    for feature in schema.features_constraint:
        if feature.type:
            body += [[feature.name, pref + 'the feature should be of type ' + feature.type]]
        if feature.presence:
            presence = "fraction of " + str(feature.presence) if feature.presence <= 1 else "number of " + str(
                feature.presence)
            body += [[feature.name, pref + 'the feature should be present with a minimum ' + presence]]
        if feature.distinctness:
            if feature.distinctness.condition == "eq":
                body += [[feature.name,
                          pref + 'the feature distinctness should be equal to {}'.format(feature.distinctness.value)]]
            elif feature.distinctness.condition == "gt":
                body += [[feature.name,
                          pref + 'the feature distinctness should be below {}'.format(feature.distinctness.value)]]
            elif feature.distinctness.condition == "lt":
                body += [[feature.name,
                          pref + 'the feature distinctness should be over {}'.format(feature.distinctness.value)]]
        if feature.domain:
            if isinstance(feature.domain, list):
                body += [[feature.name, pref + 'the feature values should be in {}'.format(feature.domain)]]
            else:
                if hasattr(feature.domain, 'min'):
                    body += [[feature.name,
                              pref + 'the feature values should be greater than {}'.format(feature.domain.min)]]
                if hasattr(feature.domain, 'max'):
                    body += [
                        [feature.name, pref + 'the feature values should be less than {}'.format(feature.domain.max)]]
        if feature.regex:
            body += [[feature.name, pref + 'the feature values should match the pattern {}'.format(feature.regex)]]
        if feature.drift:
            body += [[feature.name,
                      pref + 'the drift between the current dataset and the previous should be less than {}'.format(
                          feature.drift)]]
        if feature.outliers:
            body += [[feature.name,
                      pref + 'the feature values should be in the range [mean-{0}*std,mean+{0}*std]'.format(
                          feature.outliers)]]
        if feature.highest_frequency_threshold:
            threshold = feature.highest_frequency_threshold
            message = str(threshold) + " rows" if threshold > 1 else str(threshold)
            body += [[feature.name, pref + 'the frequency of the most present value within the feature {} should be'
                                           ' less or equal to {}'.format(feature.name, message)]]
        if feature.mapped_to_only_one:
            body += [[feature.name, f'{pref}{feature.name} values should take the same value for the following features'
                                    f' {feature.mapped_to_only_one}']]

    if hasattr(schema, "slices_constraint"):
        for slices in schema.slices_constraint:
            new_pref = "for " + slices.slicing_column
            if slices.values_to_take:
                new_pref = new_pref + " in " + str(slices.values_to_take)
                if slices.values_to_drop:
                    new_pref = new_pref + " and not in " + str(slices.values_to_drop)
            if slices.values_to_drop:
                new_pref = new_pref + " not in " + str(slices.values_to_drop)
            new_pref = new_pref + ": "
            new_pref = pref + "; " + new_pref if pref else new_pref
            _get_constraints_list_schema(slices.schema, body, new_pref)


def _is_percentage(data: pd.DataFrame, feature: str) -> bool:
    """
    a function that uses the feature name to guess if the feature is an id
    :param data: df with data
    :param feature: feature name
    :return: True if the feature is an id and False otherwise
    """
    return re.match('.*([^a-zA-Z]|^)(percent|percentage)([^a-zA-Z]|$).*', feature.lower()) is not None \
           and data[feature].max() <= 100


def _infer_domain(data: pd.DataFrame, feature: str, max_domain_values: int) -> object:
    """
    a function to infer a given feature domain (possible values for str features or min, max for numerical ones)
    :param data: df with the data 
    :param feature: feature name
    :param max_domain_values: max number of unique values within a feature to infer a str domain
    :return: a corresponding domain
    """
    if _get_feature_type(data, feature) == "str":
        return [str(x) for x in data[feature].unique() if x == x] if len(
            data[feature].unique()) < max_domain_values else ""
    else:
        has_domain = False
        domain = SimpleNamespace()
        if min(data[feature]) >= 0:
            domain.min = 0
            has_domain = True
        if max(data[feature]) <= 100 and _is_percentage(data, feature):
            domain.max = 100
            has_domain = True
        if not has_domain:
            domain = ""
        return domain


def infer_schema(data: pd.DataFrame = None, data_path: str = '', delimiter: str = ',',
                 max_domain_values: int = 20) -> SimpleNamespace:
    """
    A function to infer a schema from data
    :param data: data to generate schema
    :param data_path: path to data to generate the schema
    :param delimiter: data file delimiter
    :param max_domain_values: max number of unique values within a feature to infer a str domain
    :return: a schema for the given data
    """
    if data is None and data_path is None:
        raise ValueError('data and data_path are null')
    if data_path:
        data = pd.read_csv(data_path, delimiter=delimiter)
    data_size = len(data)

    schema = SimpleNamespace()
    schema.dataset_constraint = SimpleNamespace()
    schema.dataset_constraint.min_fraction_threshold = ""
    schema.dataset_constraint.max_fraction_threshold = ""
    schema.dataset_constraint.unicity_features = ""
    schema.dataset_constraint.batch_id_column = ""
    schema.dataset_constraint.joins_constraint = []
    schema.features_constraint = []
    for feature in data.columns:
        presence = data.describe(include='all').loc['count', feature] / data_size
        add_feature_constraint(schema=schema, name=feature, feature_type=_get_feature_type(data, feature),
                               presence=1 if presence == 1 else presence - 0.2 if presence > 0.2 else "",
                               domain=_infer_domain(data, feature, max_domain_values))
    schema.slices_constraint = []
    schema.custom_constraint = SimpleNamespace()

    return schema


def get_compact_schema(schema: SimpleNamespace = None, schema_path: str = None) -> pd.DataFrame:
    """
    a function that returns a constraints df from a schema
    :param schema: schema with the constraints
    :param schema_path: path to the schema with the constraints
    :return: a df with the applied constraints
    """
    if schema is None and schema_path is None:
        raise ValueError('schema and schema_path are null')
    if schema_path:
        schema = json.load(open('file.json'), object_hook=lambda d: SimpleNamespace(**d))
    cols = ['element', 'constraint']
    body = []
    _get_constraints_list_schema(schema, body)
    df = pd.DataFrame(data=body, columns=cols)
    return df


def set_min_fraction_threshold(schema: SimpleNamespace, threshold: float) -> SimpleNamespace:
    """
    set a min fraction threshold
    :param schema: schema to be modified
    :param threshold: the desired threshold
    :return: modified schema
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError('threshold should be an integer or a float')
    schema.dataset_constraint.min_fraction_threshold = threshold
    return schema


def set_max_fraction_threshold(schema: SimpleNamespace, threshold: float) -> SimpleNamespace:
    """
    set a max fraction threshold
    :param schema: schema to be modified
    :param threshold: the desired threshold
    :return: modified schema
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError('threshold should be an integer or a float')
    schema.dataset_constraint.max_fraction_threshold = threshold
    return schema


def set_unicity_features(schema: SimpleNamespace, features: Union[str, list]) -> SimpleNamespace:
    """
    set unicity features
    :param schema: schema to be modified
    :param features: features that act as a row identifier
    :return: modified schema
    """
    if not isinstance(features, (list, str)):
        raise TypeError('features should be the name of a feature or a list of feature names')
    schema.dataset_constraint.unicity_features = features
    return schema


def set_batch_id_column(schema: SimpleNamespace, feature: str) -> SimpleNamespace:
    """
        set batch id column feature
        :param schema: schema to be modified
        :param feature: feature that contain batch id
        :return: modified schema
    """
    schema.dataset_constraint.batch_id_column = feature
    return schema


def add_feature_constraint(schema: SimpleNamespace, name: str, feature_type: str, presence: float = None,
                           distinctness_condition: str = None, distinctness_value: float = None,
                           domain: SimpleNamespace = None, str_domain: list = None, min_value: float = None,
                           max_value: float = None, regex: str = None, outliers: float = None,
                           highest_frequency_threshold: float = None, drift: float = None,
                           mapped_to_only_one: object = None) -> SimpleNamespace:
    """
    add a feature to the schema features constraints
    :param drift:
    :param schema: schema to modify
    :param name: feature name
    :param feature_type: feature type either "str", "int" or "float"
    :param presence: feature presence
    :param distinctness_condition: feature distinctness condition either "gt" "eq" or "lt
    :param distinctness_value: feature distinctness value
    :param domain: SimpleNameSpace with the defined domain of the feature
    :param str_domain: list of possible values
    :param min_value: min possible value
    :param max_value: max possible value
    :param regex: feature regex
    :param outliers: the k of the range [mean + k*std , mean - k*std] that define the range of non outliers values
    :param highest_frequency_threshold: highest frequency that a value could have in a column
    :param drift : drift between two batches of data
    :param mapped_to_only_one: the list of features that take only one value for a given feature value
    :return: modified schema with the new feature
    """
    if str_domain is None:
        str_domain = []
    if not isinstance(name, str):
        raise TypeError('name should be a str')
    if feature_type not in ['str', 'int', 'float']:
        raise ValueError('type should be "str", "float" or "int"')
    if presence and not isinstance(presence, (int, float)):
        raise TypeError('presence should be an integer or a float value')
    if presence and presence < 0:
        raise ValueError('presence should be a positive value')
    if distinctness_condition and distinctness_condition not in ['eq', 'gt', 'lt']:
        raise ValueError('type should be "eq", "gt" or "lt"')
    if distinctness_value and (distinctness_value < 0 or distinctness_value > 1):
        raise ValueError('distinctness_value should be between 0 and 1')
    if bool(distinctness_condition) ^ bool(distinctness_value):
        raise ValueError(
            'both distinctness_value and distinctness_condition should be present to define distinctness test')
    if domain != "" and domain is not None and not isinstance(domain, (SimpleNamespace, list)):
        raise TypeError('domain should be a SimpleNameSpace or a list')
    if domain and (str_domain or max_value or min_value):
        raise ValueError("only one of the domain, str_domain and (max_value, min_value) can be defined")
    if str_domain and not isinstance(str_domain, list):
        raise TypeError('domain should be a list or a min max dict')
    if str_domain and feature_type != 'str':
        raise TypeError('str domain cannot be defined to numerical features')
    if (max_value is not None or min_value is not None) and feature_type == 'str':
        raise TypeError('max_value and min_value cannot be defined for a str feature')
    if (not isinstance(max_value, (int, float)) and max_value != "" and max_value is not None) or \
            (not isinstance(min_value, (int, float)) and min_value != "" and min_value is not None):
        raise ValueError('max_value and min_value should be numbers')
    if max_value is not None and min_value is not None and "" not in [max_value, min_value] and max_value <= min_value:
        raise ValueError('max_value should be greater than min_value')
    if str_domain and (max_value or min_value):
        raise ValueError('you cannot affect a str domain and a min/max value to the same feature')
    if regex and not isinstance(regex, str):
        raise TypeError('regex should be a regex str')
    if outliers and not isinstance(outliers, (int, float)):
        raise TypeError('outliers should be a float')
    if outliers and feature_type == "str":
        raise ValueError('outliers cannot be defined for a str feature')
    if highest_frequency_threshold and (
            not isinstance(highest_frequency_threshold, (float, int)) or highest_frequency_threshold < 0):
        raise TypeError('highest_frequency_threshold should be a positive float')
    if drift is not None:
        if not isinstance(drift, (int, float)):
            raise TypeError('drift should be a float')
        if drift < 0 or drift > 1:
            raise ValueError('drift should be between 0 and 1')
    if mapped_to_only_one and not isinstance(mapped_to_only_one, (list, str)):
        raise TypeError('mapped_to_only_one should be the name of a feature or a list of feature names')

    if name in [feature.name for feature in schema.features_constraint]:
        raise ValueError('{} is already in the schema'.format(name))

    feature = SimpleNamespace()
    feature.name = name
    feature.type = feature_type
    feature.presence = presence if presence else ""
    if distinctness_condition:
        feature.distinctness = SimpleNamespace()
        feature.distinctness.condition = distinctness_condition
        feature.distinctness.value = distinctness_value
    else:
        feature.distinctness = ""
    feature.domain = domain if domain else str_domain if str_domain else ""
    if min_value is not None or max_value is not None:
        feature.domain = SimpleNamespace()
    if min_value is not None:
        feature.domain.min = min_value
    if max_value is not None:
        feature.domain.max = max_value

    feature.regex = regex if regex else ""
    feature.drift = drift if drift else ""
    feature.outliers = outliers if outliers else ""
    feature.highest_frequency_threshold = highest_frequency_threshold if highest_frequency_threshold else ""
    feature.mapped_to_only_one = mapped_to_only_one if mapped_to_only_one else ""
    schema.features_constraint += [feature]
    return schema


def add_join_constraints(schema: SimpleNamespace, name: str, left_on: str, right_on: str,
                         threshold: float) -> SimpleNamespace:
    """
    add a join constraint to the schema
    :param schema: schema to modify
    :param name: join name
    :param left_on: left table join feature
    :param right_on: right table join feature
    :param threshold: minimum join quality accepted
    :return: modified schema
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError('threshold should be an integer or a float')
    if threshold <= 0 or threshold >= 1:
        raise ValueError('threshold should be between 0 and 1')

    join_constraint = SimpleNamespace()
    join_constraint.name = name
    join_constraint.left_on = left_on
    join_constraint.right_on = right_on
    join_constraint.threshold = threshold
    for current_join_constraint in schema.dataset_constraint.joins_constraint:
        if join_constraint.name == current_join_constraint.name:
            raise ValueError('constraint name exists already')
    schema.dataset_constraint.joins_constraint += [join_constraint]
    return schema


def add_slice_constraint(schema: SimpleNamespace, slicing_column: str, values_to_take: list = None,
                         values_to_drop: list = None, inner_schema: SimpleNamespace = None) -> SimpleNamespace:
    """
    add a slice constraint to a schema
    :param schema: schema to modify
    :param slicing_column: column used to slice data
    :param values_to_take: slicing_column values to keep in the slice to validate
    :param values_to_drop: slicing_column values to drop from the slice to validate
    :param inner_schema: the slice's schema
    :return: a modified schema
    """
    if not isinstance(inner_schema, SimpleNamespace):
        raise TypeError('inner_schema should be a SimpleNameSpace schema')
    if not isinstance(slicing_column, str):
        raise TypeError('slicing_column should be a str')
    if values_to_take is not None and not isinstance(values_to_take, list):
        raise TypeError('values_to_take should be a list of features')
    if values_to_drop is not None and not isinstance(values_to_drop, list):
        raise TypeError('values_to_drop should be a list of features')
    if not values_to_drop and not values_to_take:
        raise ValueError('at least one of values_to_drop or values_to_drop should be an non-empty values list')
    if values_to_drop and values_to_take and len([value for value in values_to_take if value in values_to_drop]) > 0:
        raise ValueError('the intersection between values_to_drop and values_to_take should be empty')

    # test the slice is not already defined
    try:
        get_slice_constraint(schema, slicing_column, values_to_take, values_to_drop)
    except KeyError:
        slice_constraint = SimpleNamespace()
        slice_constraint.slicing_column = slicing_column
        slice_constraint.values_to_take = values_to_take if values_to_take else []
        slice_constraint.values_to_drop = values_to_drop if values_to_drop else []
        slice_constraint.schema = inner_schema
        schema.slices_constraint += [slice_constraint]
        return schema

    raise KeyError('slice already exists, try modifying it using get_slice_constraint()')


def add_custom_constraint(schema: SimpleNamespace, key: str, value: object) -> SimpleNamespace:
    """
    add a custom constraint to the custom constraints attribute
    :param schema: schema to modify
    :param key: name of the attribute to add
    :param value: the key value
    :return: a modified schema
    """
    custom_constraint = schema.custom_constraint
    setattr(custom_constraint, key, value)
    return schema


def remove_join_constraint(schema: SimpleNamespace, join_name: str) -> SimpleNamespace:
    """
    remove a join constraint from a schema
    :param schema: schema to modify
    :param join_name: name of the join to remove
    :return: a modified schema
    """
    for join in schema.dataset_constraint.joins_constraint:
        if join.name == join_name:
            schema.dataset_constraint.joins_constraint.remove(join)
            return schema
    raise KeyError('"{}" not found in the join constraints list'.format(join_name))


def remove_feature_constraint(schema: SimpleNamespace, feature_name: str) -> SimpleNamespace:
    """
    remove a feature from the schema
    :param schema: schema to modify
    :param feature_name: name of the feature to delete
    :return: modified schema
    """
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            schema.features_constraint.remove(feature)
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def remove_slice_constraint(schema: SimpleNamespace, slicing_column: str, values_to_take: list = None,
                            values_to_drop: list = None) -> object:
    """
    remove a slice constraint from the schema
    :param schema: schema to modify
    :param slicing_column: column used to slice data
    :param values_to_take: slicing_column values to keep in the slice to validate
    :param values_to_drop: slicing_column values to drop from the slice to validate
    :return: a modified schema
    """
    if values_to_drop is None:
        values_to_drop = []
    if values_to_take is None:
        values_to_take = []
    if not isinstance(slicing_column, str):
        raise TypeError('slicing_column should be a str')
    if not isinstance(values_to_take, list):
        raise TypeError('values_to_take should be a list of features')
    if not isinstance(values_to_drop, list):
        raise TypeError('values_to_drop should be a list of features')
    if not values_to_take and not values_to_drop:
        raise ValueError('at least one of values_to_drop or values_to_drop should be an non-empty values list')

    for slice_constraint in schema.slices_constraint:
        if slice_constraint.slicing_column == slicing_column and \
                slice_constraint.values_to_take == values_to_take and \
                slice_constraint.values_to_drop == values_to_drop:
            schema.slices_constraint.remove(slice_constraint)
            return schema
    raise KeyError('slice is not in the schema')


def set_type(schema: SimpleNamespace, feature_name: str, feature_type: str) -> SimpleNamespace:
    """
    set a feature type
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param feature_type: desired type
    :return: modified schema
    """
    if not isinstance(feature_name, str):
        raise TypeError('name should be a str')
    if feature_type not in ['str', 'int', 'float']:
        raise ValueError('type should be "str", "float" or "int"')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            if feature.type in ['int', 'float'] and feature_type == 'str':
                feature.domain = ""
            feature.type = feature_type
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_presence(schema: SimpleNamespace, feature_name: str, presence: float) -> SimpleNamespace:
    """
    set a feature presence
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param presence: desired presence
    :return: modified schema
    """
    if presence and not isinstance(presence, (int, float)):
        raise TypeError('presence should be an integer or a float value')
    if presence < 0:
        raise ValueError('presence should be a positive value')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.presence = presence
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_distinctness(schema: SimpleNamespace, feature_name: str, distinctness_condition: str,
                     distinctness_value: float) -> object:
    """
    set a feature distinctness
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param distinctness_condition: distinctness condition. either "gt", "eq" or "lt"
    :param distinctness_value: distinctness value
    :return: modified schema
    """
    if distinctness_condition not in ['eq', 'gt', 'lt']:
        raise ValueError('type should be "eq", "gt" or "lt"')
    if distinctness_value < 0 or distinctness_value > 1:
        raise ValueError('distinctness_value should be between 0 and 1')

    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.distinctness = SimpleNamespace()
            feature.distinctness.condition = distinctness_condition
            feature.distinctness.value = distinctness_value
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_domain(schema: SimpleNamespace, feature_name: str, domain: str = None, max_value: float = None,
               min_value: float = None) -> object:
    """
    set a feature domain
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param domain: list of possible values for str features or empty string to clear it
    :param max_value: max possible value for numerical features
    :param min_value: min possible value for numerical features
    :return: modified schema
    """
    if domain and not isinstance(domain, list):
        raise TypeError('domain should be a list')
    if max_value is not None and not isinstance(max_value, (int, float)) and max_value != "":
        raise TypeError('max_value should be a float or an empty string')
    if min_value is not None and not isinstance(min_value, (int, float)) and min_value != "":
        raise TypeError('min_value should be a float or an empty string')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            if min_value is not None and max_value is not None:
                if min_value != "" and max_value != "" and min_value > max_value:
                    raise ValueError('min_value should be less than max_value')
            if (min_value is not None and min_value != "") or (max_value is not None and max_value != ""):
                if feature.type == "str":
                    raise ValueError('max_value and min_value cannot be defined for a str feature')
                feature.domain = SimpleNamespace()
            if domain or domain == "":
                feature.domain = domain
            if max_value is not None:
                feature.domain.max = max_value
            if min_value is not None:
                feature.domain.min = min_value
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_regex(schema: SimpleNamespace, feature_name: str, regex: str) -> SimpleNamespace:
    """
    set a feature presence
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param regex: desired regex
    :return: modified schema
    """
    if regex and not isinstance(regex, str):
        raise TypeError('regex should be a regex str')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.regex = regex
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_drift(schema: SimpleNamespace, feature_name: str, drift: float) -> SimpleNamespace:
    """
    set a feature max drift
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param drift: drift value
    :return: modified schema
    """
    if drift and (not isinstance(drift, (float, int)) or drift < 0):
        raise TypeError('drift should be a positive float')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.drift = drift
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_outliers(schema, feature_name, k):
    """
    set a feature to change the k value for the range of non anomaly values [mean + k*std, mean - k*std]
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param k: desired k
    :return: modified schema
    """
    if k and not isinstance(k, (int, float)):
        raise TypeError('k value should be a float')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            if k and feature.type == "str":
                raise ValueError('outliers cannot be defined for a str feature')
            feature.outliers = k
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_highest_frequency_threshold(schema: SimpleNamespace, feature_name: str,
                                    highest_frequency_threshold: float) -> SimpleNamespace:
    """
    set a feature max frequency for a given value within a column
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param highest_frequency_threshold: desired highest frequency threshold allowed
    :return: modified schema
    """
    if highest_frequency_threshold and (
            not isinstance(highest_frequency_threshold, (float, int)) or highest_frequency_threshold < 0):
        raise TypeError('highest_frequency_threshold should be a positive float')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.highest_frequency_threshold = highest_frequency_threshold
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def set_mapped_to_only_one(schema: SimpleNamespace, feature_name: str, mapped_to_only_one: object) -> SimpleNamespace:
    """
    set the features that can take only one value for a give value of feature 'feature_name'
    :param schema: schema to modify
    :param feature_name: name of the feature to modify
    :param mapped_to_only_one: mapped feature / features
    :return: modified schema
    """
    if mapped_to_only_one and not isinstance(mapped_to_only_one, (list, str)):
        raise TypeError('mapped_to_only_one should be the name of a feature or a list of feature names')
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            feature.mapped_to_only_one = mapped_to_only_one
            return schema
    raise ValueError('{} feature is not in the schema'.format(feature_name))


def get_min_fraction_threshold(schema):
    return schema.dataset_constraint.min_fraction_threshold


def get_max_fraction_threshold(schema):
    return schema.dataset_constraint.max_fraction_threshold


def get_unicity_features(schema):
    return schema.dataset_constraint.unicity_features


def get_all_joins_constraint(schema):
    return schema.dataset_constraint.joins_constraint


def get_all_features_constraint(schema):
    return schema.features_constraint


def get_join_constraint(schema, join_name):
    for join in schema.dataset_constraint.joins_constraint:
        if join.name == join_name:
            return join
    raise KeyError('"{}" not found in the joins constraint list'.format(join_name))


def get_feature_constraint(schema, feature_name):
    for feature in schema.features_constraint:
        if feature.name == feature_name:
            return feature
    raise KeyError('"{}" not found in the features constraint list'.format(feature_name))


def get_slice_constraint(schema, slicing_column, values_to_take=None, values_to_drop=None):
    if values_to_drop is None:
        values_to_drop = []
    if values_to_take is None:
        values_to_take = []
    if not isinstance(slicing_column, str):
        raise TypeError('slicing_column should be a str')
    if not isinstance(values_to_take, list):
        raise TypeError('values_to_take should be a list of features')
    if not isinstance(values_to_drop, list):
        raise TypeError('values_to_drop should be a list of features')
    if not values_to_take and not values_to_drop:
        raise ValueError('at least one of values_to_drop or values_to_drop should be an non-empty values list')

    for slice_constraint in schema.slices_constraint:
        if slice_constraint.slicing_column == slicing_column and \
                slice_constraint.values_to_take == values_to_take and \
                slice_constraint.values_to_drop == values_to_drop:
            return slice_constraint
    raise KeyError('slice is not in the schema')


def get_feature_type(schema, feature_name):
    return get_feature_constraint(schema, feature_name).type


def get_feature_presence(schema, feature_name):
    return get_feature_constraint(schema, feature_name).presence


def get_feature_distinctness(schema, feature_name):
    return get_feature_constraint(schema, feature_name).distinctness


def get_feature_domain(schema, feature_name):
    return get_feature_constraint(schema, feature_name).domain


def get_feature_min(schema, feature_name):
    return get_feature_constraint(schema, feature_name).domain.min


def get_feature_max(schema, feature_name):
    return get_feature_constraint(schema, feature_name).domain.max


def get_feature_regex(schema, feature_name):
    return get_feature_constraint(schema, feature_name).regex


def get_feature_drift(schema, feature_name):
    return get_feature_constraint(schema, feature_name).drift


def get_feature_outliers(schema, feature_name):
    return get_feature_constraint(schema, feature_name).outliers


def get_feature_highest_frequency_threshold(schema, feature_name):
    return get_feature_constraint(schema, feature_name).highest_frequency_threshold


def get_feature_mapped_to_only_one(schema, feature_name):
    return get_feature_constraint(schema, feature_name).mapped_to_only_one
