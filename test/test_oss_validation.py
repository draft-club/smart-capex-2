import unittest
from unittest.mock import patch
import pandas as pd
from datetime import date
from src.oss_validation import OssValidation as Validation
import json
from mock.mock import mock_open as mock_open_callable
from types import SimpleNamespace
from pandas.testing import assert_frame_equal


class TestValidateData(unittest.TestCase):

    def assertDataframeEqual(self, a, b, msg):
        # a class method to assert equality between DataFrames
        try:
            assert_frame_equal(a, b, check_like=True)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        # add equality function for dataframes
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    @patch('src.validation.Validation._get_join_table')
    @patch('src.validation.Validation.load_schema')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('builtins.open')
    @patch('json.load')
    @patch('json.dump')
    def test_validate_data(self, mock_json_dump, mock_json_load, mock_open, mock_to_csv, mock_read_csv, mock_schema,
                           mock_get_join_table):
        """
        Test the function that validate a given Dataframe / csv file.
        """

        # GIVEN :
        # The Following config
        schema = {
            "dataset_constraint": {
                "min_fraction_threshold": 0.9,
                "max_fraction_threshold": 1.2,
                "unicity_features": ["distinctness", "drift"],
                "joins_constraint": [
                    {
                        "name": "join",
                        "left_on": "cell_name",
                        "right_on": "cell",
                        "threshold": 0.99
                    }
                ],
                "batch_id_column": "date"
            },
            "custom_constraint": {
                "max_new_cells": 2,
                "max_disappeared_cells": 2,
                "max_cell_per_site": "",
                "cell_occupation_ul_percentage_presence": "",
                "cell_name_feature": "cell_name"
            },
            "features_constraint": [
                {
                    "name": "type",
                    "type": "int",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "to_exclude",
                    "type": "int",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "presence",
                    "type": "float",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "distinctness",
                    "type": "int",
                    "presence": 1,
                    "distinctness": {"condition": "eq", "value": 0},
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                }, {
                    "name": "num_domain",
                    "type": "int",
                    "presence": 1,
                    "distinctness": {"condition": "eq", "value": 1},
                    "domain": {"min": 0, "max": 100},
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                }, {
                    "name": "str_domain",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": ['a', 'b', 'c', 'd', 'e'],
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "regex",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "^regex#cell_name[2:4]##str_domain#$",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "drift",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": 0.1,
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "outliers",
                    "type": "int",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": 3,
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "cell_name",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": "site_name"
                },
                {
                    "name": "frequency",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": .5,
                    "mapped_to_only_one": ""
                },
                {
                    "name": "site_name",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                },
                {
                    "name": "date",
                    "type": "str",
                    "presence": 1,
                    "distinctness": "",
                    "domain": "",
                    "regex": "",
                    "drift": "",
                    "outliers": "",
                    "highest_frequency_threshold": "",
                    "mapped_to_only_one": ""
                }
            ],
            "slices_constraint": [
                {
                    "slicing_column": "drift",
                    "values_to_take": [],
                    "values_to_drop": ["pos"],
                    "schema": {
                        "features_constraint": [
                            {
                                "name": "presence",
                                "type": "",
                                "presence": 1,
                                "distinctness": "",
                                "domain": "",
                                "regex": "",
                                "drift": "",
                                "outliers": "",
                                "highest_frequency_threshold": "",
                                "mapped_to_only_one": ""
                            }
                        ]
                    }
                }
            ]
        }
        schema = json.loads(str(schema).replace("'", '"'), object_hook=lambda d: SimpleNamespace(**d))
        time_series_history = {'type': {"cell1": {"mean": 2.0, "squared_sum": 245, "nbr": 11, "std": 3.2},
                                        "cell2": {"mean": 2.4, "squared_sum": 252, "nbr": 11, "std": 3.1},
                                        "cell3": {"mean": 1.7, "squared_sum": 248, "nbr": 11, "std": 2.7},
                                        "cell4": {"mean": 2.2, "squared_sum": 254, "nbr": 11, "std": 4.0}},
                               'outliers': {"cell1": {"mean": 2.0, "squared_sum": 245, "nbr": 11, "std": 3.2},
                                            "cell2": {"mean": 2.4, "squared_sum": 252, "nbr": 11, "std": 3.1},
                                            "cell3": {"mean": 1.7, "squared_sum": 248, "nbr": 11, "std": 2.7},
                                            "cell4": {"mean": 2.2, "squared_sum": 254, "nbr": 11, "std": 4.0}},
                               }
        previous_data = pd.DataFrame({'type': [1, 2, 3, 4],
                                      'presence': [1.2, 2, 3, 4],
                                      'distinctness': [1, 3, 2, 1],
                                      'num_domain': [1, 2, 3, 4],
                                      'str_domain': ['a', 'b', 'c', 'd'],
                                      'regex': ['regex1a', 'regex3b', 'regex2c', 'regex1d'],
                                      'drift': ['pos', 'neg', 'pos', 'neg'],
                                      'outliers': [1, 2, 3, 4],
                                      'cell_name': ["cell1", "cell2", "cell3", "cell4"],
                                      'frequency': ['a', 'a', 'b', 'b'],
                                      'site_name': ["site1", "site2", "site1", "site2"],
                                      'date': ["date1", "date1", "date1", "date1"],
                                      'to_exclude': ['a', 'a', 'b', 'b']
                                      })
        joined_with_table = pd.DataFrame({'feature1': ['neg', 'neg', 'pos', 'neg', 'neg'],
                                          'feature2': [1, 22122012, 3, 4, 5],
                                          'cell': ["FN_C2", "FN_C3", "FN_C4", "FN_C5", "FN_C6"],
                                          })
        expected_anomalies = pd.DataFrame({
            "feature_name": ["type", "num_domain", "str_domain", "presence", "drift", "dataset anomaly", "distinctness",
                             "regex",
                             #"outliers",
                             "presence", "dataset anomaly", "frequency", "cell_name",
                             "dataset anomaly"],
            "date": [date.today()] * 13,
            "batch_id": ["date2"] * 13,
            "anomaly_short_description": ["Type anomaly", "Out-of-range values",
                                          "Unexpected string values",
                                          "Presence anomaly", "Drift anomaly",
                                          "Size anomaly", "Distinctness anomaly", "Pattern anomaly",
                                          #"Outlier anomaly",
                                          "Presence anomaly", "Duplicates anomaly", "Too frequent value anomaly",
                                          "Mapping anomaly", "Join anomaly"],
            "anomaly_long_description": ["Expected data of type: int but got float", "Unexpectedly small value: -1",
                                         "Examples contain values missing from the schema: 'letter'",
                                         "The feature was present in fewer examples than expected: "
                                         "minimum = 1, actual = 0.8",
                                         "The Linfty distance between the two batches is 0.3, "
                                         "above the threshold 0.1. "
                                         "The feature value with maximum difference is: neg",
                                         "The ratio of num examples in the current dataset versus the previous span is "
                                         "1.25, which is above the threshold 1.2",
                                         "Actual distinctness value: 0.4 is not equal threshold: 0",
                                         "At least one value: regex1a does NOT match pattern : "
                                         "^regex#cell_name[2:4]##str_domain#$",
                                         #"Value 22122012.0 for cell cell2 is considered as an anomaly",
                                         "Sliced by drift: The feature was present in fewer examples than expected: "
                                         "minimum = 1, actual = 0.75",
                                         "There is duplicates in the dataset, the features used to check are "
                                         "['distinctness', 'drift']",
                                         'The value "a" represent a ratio of 0.6 of non missing data which is over '
                                         'threshold: 0.5',
                                         'At least one value (FN_C1) is mapped to multiple site_name values',
                                         'The quality of the join between the dataset and the "join" dataset is 0.75, '
                                         'which is below the threshold 0.99'
                                         ],
            "examples": [[3.2], [-1], ['letter'], None, None, None, [1, 2], ['regex1a', 'regex1e'],
                         #[{'cell_name': 'cell2', 'outliers': 22122012}],
                         {'feature': 'drift', 'values_to_take': [], 'values_to_drop': ['pos'], 'examples': None},
                         [{'distinctness': 1, 'drift': 'neg'}], ['a'], ['FN_C1'], None
                         ],
            "type": ["oss"] * 13
        })
        # and the following data input
        input_data = pd.DataFrame({'type': [1, 2, 3, 4, 3.2],
                                   'presence': [1, None, 3, 4, 5],
                                   'distinctness': [1, 1, 1, 1, 2],
                                   'num_domain': [1, 2, 3, 4, -1],
                                   'str_domain': ['a', 'b', 'c', 'd', 'letter'],
                                   'regex': ['regex1a', 'regex1e', 'regex_C3c', 'regex_C4d', 'regex_C1letter'],
                                   'drift': ['neg', 'neg', 'pos', 'neg', 'neg'],
                                   'outliers': [1, 22122012, 3, 4, 5],
                                   'cell_name': ["FN_C1", "FN_C2", "FN_C3", "FN_C4", "FN_C1"],
                                   'frequency': ['a', 'a', 'a', 'b', 'b'],
                                   'site_name': ["site1", "site2", "site1", "site2", "site2"],
                                   'date': ["date2", "date2", "date2", "date2", "date2"],
                                   'to_exclude': ['a', 'a', 'a', 'a', 'a']
                                   })

        # and patching the IO functions to directly provide objects from the config above
        mock_schema.return_value = schema
        mock_open.new_callable = mock_open_callable
        mock_json_load.return_value = time_series_history
        mock_get_join_table.return_value = joined_with_table

        # WHEN :
        # a user validate data
        validation_tool = Validation('oss', paths={'join': {'path': 'test/class_test.csv', 'delimiter': '!'}})
        output_anomalies = validation_tool.validate_data(data=input_data, previous_data=previous_data,
                                                         exclude=["to_exclude"])
        # THEN:
        # the output anomaly report should be equal to the expected anomaly report
        self.assertEqual(output_anomalies.sort_values(['feature_name', 'anomaly_short_description']).
                         reset_index(drop=True).sort_index(axis=1),
                         expected_anomalies.sort_values(['feature_name', 'anomaly_short_description']).
                         reset_index(drop=True).sort_index(axis=1))

    @patch('src.validation.Validation.load_schema')
    def test_get_anomalies_examples(self, mock_schema):
        # GIVEN :
        # The Following input
        anomalies = pd.DataFrame({
            "feature_name": ["type", "num_domain", "str_domain", "presence", "drift", "dataset anomaly",
                             "distinctness", "regex",
                             #"outliers",
                             "presence", "dataset anomaly", "frequency", "WBTS",
                             "dataset anomaly"],
            "date": [date.today()] * 13,
            "batch_id": ["date2"] * 13,
            "anomaly_short_description": ["Type anomaly", "Out-of-range values",
                                          "Unexpected string values",
                                          "Presence anomaly", "Drift anomaly",
                                          "Size anomaly", "Distinctness anomaly", "Pattern anomaly",
                                          #"Outlier anomaly",
                                          "Presence anomaly", "Duplicates anomaly", "Too frequent value anomaly",
                                          "Mapping anomaly", "Join anomaly"],
            "anomaly_long_description": ["Expected data of type: int but got float", "Unexpectedly small value: -1",
                                         "Examples contain values missing from the schema: 'letter'",
                                         "The feature was present in fewer examples than expected: "
                                         "minimum = 1, actual = 0.8",
                                         "The Linfty distance between the two batches is 0.3, "
                                         "above the threshold 0.1. "
                                         "The feature value with maximum difference is: neg",
                                         "The ratio of num examples in the current dataset versus the previous span is "
                                         "1.25, which is above the threshold 1.2",
                                         "Actual distinctness value: 0.4 is not equal threshold: 0",
                                         "At least one value: regex1a does NOT match pattern : "
                                         "^regex#cell_name[2:4]##str_domain#$",
                                         #"Value 22122012 for cell cell2 is considered as an anomaly",
                                         "Sliced by drift: The feature was present in fewer examples than expected: "
                                         "minimum = 1, actual = 0.75",
                                         "There is duplicates in the dataset, the features used to check are "
                                         "['distinctness', 'drift']",
                                         'The value "a" represent a ratio of 0.6 of non missing data which is over '
                                         'threshold: 0.5',
                                         'At least one value (FN_C1) is mapped to multiple site_name values',
                                         'The quality of the join between the dataset and the "join" dataset is 0.75, '
                                         'which is below the threshold 0.99'
                                         ],
            "examples": [[3.2], [-1], ['letter'], None, ['neg'], None, [1, 2], ['regex1a', 'regex1e'],
                         #[{'WBTS': 'cell2', 'outliers': 22122012}],
                         {'feature': 'drift', 'values_to_take': [], 'values_to_drop': ['pos'], 'examples': None},
                         [{'distinctness': 1, 'drift': 'neg'}], ['a'], ['FN_C1'], None
                         ],
            "type": ["oss"] * 13
        })
        data = pd.DataFrame({'type': [1, 2, 3, 4, 3.2],
                             'presence': [1, None, 3, 4, 5],
                             'distinctness': [1, 1, 1, 1, 2],
                             'num_domain': [1, 2, 3, 4, -1],
                             'str_domain': ['letter', 'b', 'c', 'd', 'e'],
                             'regex': ['regex1a', 'regex1b', 'regex1c', 'regex1d', 'regex1e'],
                             'drift': ['neg', 'neg', 'pos', 'neg', 'neg'],
                             'outliers': [1, 22122012, 3, 4, 5],
                             'WBTS': ["cell1", "cell2", "cell3", "cell4", "cell1"],
                             'frequency': ['a', 'a', 'a', 'b', 'b'],
                             'site_name': ["site1", "site2", "site1", "site2", "site2"],
                             'date': ["date2", "date2", "date2", "date2", "date2"]
                             })
        # and the following configuration
        expected_report = anomalies.sort_values(by=['feature_name', 'anomaly_short_description']).reset_index(drop=True)
        expected_report['id'] = expected_report['type'] + '-' + expected_report['batch_id'] + '-' + \
                                expected_report['date'].map(str) + '-' + pd.Series(range(len(anomalies))).map(str)
        expected_report = expected_report.set_index('id')

        expected_examples = pd.concat([data.iloc[[4], :], data.iloc[[0], :], data.iloc[[3], :], data.iloc[[1], :],
                                       data.iloc[[0], :], data.iloc[[3], :], data.iloc[[0], :], data.iloc[[4], :],
                                       data.iloc[[2], :], data.iloc[[1], :], data.iloc[[4], :], data.iloc[[3], :],
                                       data.iloc[[1], :], data.iloc[[0], :], data.iloc[[1], :], data.iloc[[0], :],
                                       data.iloc[[2], :], data.iloc[[4], :], data.iloc[[1], :], data.iloc[[1], :],
                                       data.iloc[[1], :], data.iloc[[4], :], data.iloc[[0], :], data.iloc[[0], :],
                                       data.iloc[[4], :]])
        expected_examples['anomaly_id'] = [f'oss-date2-{date.today()}-0', f'oss-date2-{date.today()}-0',
                                           f'oss-date2-{date.today()}-1', f'oss-date2-{date.today()}-1',
                                           f'oss-date2-{date.today()}-1', f'oss-date2-{date.today()}-4',
                                           f'oss-date2-{date.today()}-4', f'oss-date2-{date.today()}-4',
                                           f'oss-date2-{date.today()}-4', f'oss-date2-{date.today()}-4',
                                           f'oss-date2-{date.today()}-5', f'oss-date2-{date.today()}-5',
                                           f'oss-date2-{date.today()}-5', f'oss-date2-{date.today()}-5',
                                           f'oss-date2-{date.today()}-6', f'oss-date2-{date.today()}-6',
                                           f'oss-date2-{date.today()}-6', f'oss-date2-{date.today()}-7',
                                           f'oss-date2-{date.today()}-8', f'oss-date2-{date.today()}-9',
                                           f'oss-date2-{date.today()}-10', f'oss-date2-{date.today()}-11',
                                           f'oss-date2-{date.today()}-11', f'oss-date2-{date.today()}-12',
                                           f'oss-date2-{date.today()}-13',
                                           ]
        expected_examples = expected_examples.sort_values(['anomaly_id', 'type']).reset_index(drop=True)
        columns_to_convert = list(expected_examples.columns)
        columns_to_convert.remove("anomaly_id")
        examples_col = list(map(str, expected_examples[columns_to_convert].to_dict(orient='records')))
        expected_examples['dv_example'] = examples_col
        expected_examples = expected_examples[['anomaly_id', 'dv_example']]

        # and patching the IO functions to directly provide objects from the config above
        mock_schema.return_value = None

        # WHEN :
        # a user call the get anomalies method
        validation_tool = Validation('oss', paths={'join': {'path': '/home/aronofsky/blackSwan.csv', 'delimiter': '!'}})
        examples, report = validation_tool.get_anomalies_examples(anomalies, data)
        examples = examples.sort_values(['anomaly_id', 'dv_example']).reset_index(drop=True)
        # THEN:
        # the output anomaly report should be equal to the expected anomaly report and
        # the output examples df should be equal to the expected examples df
        assert_frame_equal(report, expected_report, check_like=True)
        assert_frame_equal(examples, expected_examples, check_like=True)
        return examples, report


if __name__ == '__main__':
    unittest.main()
