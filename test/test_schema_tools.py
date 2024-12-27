import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
import src.schema_tools as tools
import json
from types import SimpleNamespace


class TestSchemaTools(unittest.TestCase):
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
            ]
        },
        "custom_constraint": {
            "max_new_cells": 2,
            "max_disappeared_cells": 2,
            "max_cell_per_site": "",
            "cell_occupation_ul_percentage_presence": "",
        },
        "features_constraint": [
            {
                "name": "to_test",
                "type": "str",
                "presence": 1,
                "distinctness": {"condition": "gt", "value": .45},
                "domain": ['uP', 'dOwN'],
                "regex": ".*",
                "drift": 0.2,
                "outliers": "",
                "highest_frequency_threshold": 0.8,
                "mapped_to_only_one": ['presence', 'type']
            }
        ],
        "slices_constraint": [
            {
                "slicing_column": "drift",
                "values_to_take": ["neg"],
                "values_to_drop": [],
                "schema": {
                    "features_constraint": [
                        {
                            "name": "inner_to_test",
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

    def test_get_dict(self):
        # GIVEN :
        # The Following config
        simple_name_space_schema = json.loads(str(self.schema).replace("'", '"'),
                                              object_hook=lambda d: SimpleNamespace(**d))

        # WHEN :
        # a user generate a dict from SimpleNameSpace
        regenerated_dict_schema = tools._get_dict(simple_name_space_schema)
        # THEN:
        self.assertEqual(regenerated_dict_schema, self.schema)

    def test_get_remove_add(self):
        # GIVEN :
        # The Following config
        schema = json.loads(str(self.schema).replace("'", '"'), object_hook=lambda d: SimpleNamespace(**d))
        # WHEN :
        # a user generate a get a join_constraints, feature_constraint or a slice_constraint
        join_constraint = tools.get_join_constraint(schema, 'join')
        feature_constraint = tools.get_feature_constraint(schema, 'to_test')
        slice_constraint = tools.get_slice_constraint(schema, "drift", ['neg'])
        # remove it from the schema by it key
        tools.remove_join_constraint(schema, 'join')
        tools.remove_feature_constraint(schema, 'to_test')
        tools.remove_slice_constraint(schema, "drift", ['neg'])
        # adding it again to the schema
        tools.add_join_constraints(schema, join_constraint.name, join_constraint.left_on, join_constraint.right_on,
                                   join_constraint.threshold)
        tools.add_feature_constraint(schema, 'to_test', feature_type=feature_constraint.type,
                                     presence=feature_constraint.presence, domain=feature_constraint.domain,
                                     distinctness_value=feature_constraint.distinctness.value,
                                     distinctness_condition=feature_constraint.distinctness.condition,
                                     regex=feature_constraint.regex, outliers=feature_constraint.outliers,
                                     highest_frequency_threshold=feature_constraint.highest_frequency_threshold,
                                     mapped_to_only_one=feature_constraint.mapped_to_only_one,
                                     drift=feature_constraint.drift
                                     )
        tools.add_slice_constraint(schema, slicing_column=slice_constraint.slicing_column,
                                   values_to_take=slice_constraint.values_to_take,
                                   values_to_drop=slice_constraint.values_to_drop,
                                   inner_schema=slice_constraint.schema
                                   )
        # THEN :
        schema_dict = tools._get_dict(schema)
        self.assertEqual(self.schema, schema_dict)

    def test_get_set(self):
        # GIVEN :
        # The Following config
        schema = json.loads(str(self.schema).replace("'", '"'), object_hook=lambda d: SimpleNamespace(**d))
        # WHEN :
        # a user gets an attribute
        type = tools.get_feature_type(schema, 'to_test')
        presence = tools.get_feature_presence(schema, 'to_test')
        distinctness = tools.get_feature_distinctness(schema, 'to_test')
        domain = tools.get_feature_domain(schema, 'to_test')
        regex = tools.get_feature_regex(schema, 'to_test')
        drift = tools.get_feature_drift(schema, 'to_test')
        outliers = tools.get_feature_outliers(schema, 'to_test')
        highest_frequency_threshold = tools.get_feature_highest_frequency_threshold(schema, 'to_test')
        mapped_to_only_one = tools.get_feature_mapped_to_only_one(schema, 'to_test')
        # and sets it afterward
        tools.set_type(schema, 'to_test', type)
        tools.set_presence(schema, 'to_test', presence)
        tools.set_distinctness(schema, 'to_test', distinctness.condition, distinctness.value)
        tools.set_domain(schema, 'to_test', domain)
        tools.set_regex(schema, 'to_test', regex)
        tools.set_drift(schema, 'to_test', drift)
        tools.set_outliers(schema, 'to_test', outliers)
        tools.set_highest_frequency_threshold(schema, 'to_test', highest_frequency_threshold)
        tools.set_mapped_to_only_one(schema, 'to_test', mapped_to_only_one)
        # THEN :
        schema_dict = tools._get_dict(schema)
        self.assertEqual(self.schema, schema_dict)

    def test_get_compact_schema(self):
        # GIVEN :
        # The Following config
        schema = json.loads(str(self.schema).replace("'", '"'), object_hook=lambda d: SimpleNamespace(**d))
        expected_output = {'element': ['dataset', 'dataset', 'dataset', 'dataset', 'to_test', 'to_test', 'to_test',
                                       'to_test', 'to_test', 'to_test', 'to_test', 'to_test', 'inner_to_test'],
                           'constraint': ['the fraction of the size of the current batch compared with the previous '
                                          'one should be above 0.9',
                                          'the fraction of the size of the current batch '
                                          'compared with the previous one should be below 1.2',
                                          "dataset's subset with only the features ['distinctness', 'drift'] should "
                                          "not contain duplicate rows",
                                          'join constraint "join" on the cell_name feature',
                                          'the feature should be of type str',
                                          'the feature should be present with a minimum fraction of 1',
                                          'the feature distinctness should be below 0.45',
                                          "the feature values should be in ['uP', 'dOwN']",
                                          'the feature values should match the pattern .*',
                                          'the drift between the current dataset and the previous should be less'
                                          ' than 0.2',
                                          'the frequency of the most present value within the feature to_test should '
                                          'be less or equal to 0.8',
                                          'to_test values should take the same value for the following features '
                                          '[\'presence\', \'type\']',
                                          "for drift in ['neg']: the feature should be present with a minimum fraction "
                                          "of 1"
                                          ]
                           }
        expected_output = pd.DataFrame(expected_output)
        # WHEN:
        # a user gets the compact schema
        df = tools.get_compact_schema(schema)
        # THEN:
        assert_frame_equal(df, expected_output)

    def test_infer_schema(self):
        # GIVEN :
        # The Following config
        input_data = pd.DataFrame({'type': [1, 2, 3, 4, 3.2],
                                   'presence': [1, None, 3, 4, 5],
                                   'sector_id': [1, 2, 3, 4, 6],
                                   'str_domain': ['e', 'b', 'c', 'c', 'e'],
                                   'occupation_percentage': [25, 99.2, 85.23, 90.26, 78.52]
                                   })
        expected_output = {'dataset_constraint': {'min_fraction_threshold': '',
                                                  'max_fraction_threshold': '',
                                                  'unicity_features': '',
                                                  'batch_id_column': '',
                                                  'joins_constraint': []},
                           'features_constraint': [{'name': 'type',
                                                    'type': 'float',
                                                    'presence': 1,
                                                    'distinctness': '',
                                                    'domain': {'min': 0},
                                                    'regex': '',
                                                    'drift': '',
                                                    'outliers': '',
                                                    'highest_frequency_threshold': '',
                                                    'mapped_to_only_one': ''},
                                                   {'name': 'presence',
                                                    'type': 'float',
                                                    'presence': 0.6000000000000001,
                                                    'distinctness': '',
                                                    'domain': {'min': 0},
                                                    'regex': '',
                                                    'drift': '',
                                                    'outliers': '',
                                                    'highest_frequency_threshold': '',
                                                    'mapped_to_only_one': ''},
                                                   {'name': 'sector_id',
                                                    'type': 'int',
                                                    'presence': 1,
                                                    'distinctness': '',
                                                    'domain': {'min': 0},
                                                    'regex': '',
                                                    'drift': '',
                                                    'outliers': '',
                                                    'highest_frequency_threshold': '',
                                                    'mapped_to_only_one': ''},
                                                   {'name': 'str_domain',
                                                    'type': 'str',
                                                    'presence': 1,
                                                    'distinctness': '',
                                                    'domain': ['e', 'b', 'c'],
                                                    'regex': '',
                                                    'drift': '',
                                                    'outliers': '',
                                                    'highest_frequency_threshold': '',
                                                    'mapped_to_only_one': ''},
                                                   {'name': 'occupation_percentage',
                                                    'type': 'float',
                                                    'presence': 1,
                                                    'distinctness': '',
                                                    'domain': {'min': 0, 'max': 100},
                                                    'regex': '',
                                                    'drift': '',
                                                    'outliers': '',
                                                    'highest_frequency_threshold': '',
                                                    'mapped_to_only_one': ''}],
                           'slices_constraint': [],
                           'custom_constraint': {}}
        # WHEN:
        # a user get an inferred schema
        output = tools.infer_schema(input_data, max_domain_values=5)
        output = tools._get_dict(output)
        assert expected_output == output


if __name__ == '__main__':
    unittest.main()
