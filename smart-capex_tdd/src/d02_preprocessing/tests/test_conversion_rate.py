"""Module to test conversion rate file"""
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import pandas.testing as pd_testing

from src.d02_preprocessing.conversion_rate import (compute_conversion_rate,
                                                   post_process_lte, pre_process_lte)
from src.d02_preprocessing.tests import data as data


class TestComputeConversionRate(unittest.TestCase):
    def setUp(self):
        self.forecast_data = pd.DataFrame({
            "cell_name": ["A", "B", "C"],
            "week_period": [202210, 202210, 202210],
            "date": ["2022-03-07", "2022-03-07", "2022-03-07"],
            "total_data_traffic_dl_gb_forecasted": [15, 20, 5],
            "total_data_traffic_ul_gb_forecasted": [15, 20, 5]
        })
        self.counter_data = pd.DataFrame({
            "cell_name": ["A", "B", "C"],
            "week_period": [202110, 202110, 202110],
            "date": ["2021-03-08", "2021-03-08", "2021-03-08"],
            "total_data_traffic_dl_gb_histo": [10, 20, 10],
            "total_data_traffic_ul_gb_histo": [10, 20, 10]
        })
        self.input_template_builder = pd.DataFrame({
            "CELL": ["A", "B", "C"],
            "CODE_ELT_CELL": [1, 2, 3]
        })
        self.addTypeEqualityFunc(pd.DataFrame, self._assert_data_frame_equal)

    def _assert_data_frame_equal(self, df_1, df_2, msg):
        """
        Internal Test Case function adding Pandas Data Frame comparison to Test Case
        """
        try:
            pd_testing.assert_frame_equal(df_1, df_2)
        except AssertionError as exception:
            raise self.failureException(msg) from exception

    def test_compute_conversion_rate(self):
        # WHEN
        forecasted_data = self.forecast_data
        counter_data = self.counter_data
        input_template_builder = self.input_template_builder

        # DO
        result = compute_conversion_rate(202210, forecasted_data,
                                         counter_data, input_template_builder)

        # THEN
        expected_result = pd.DataFrame({
            "cell_name": ["A", "B", "C"],
            "traffic_dl_rate": [1.5, 1, 0.5],
            "traffic_ul_rate": [1.5, 1, 0.5],
            "week_period_forecasted": [202210, 202210, 202210]
        })
        self.assertEqual(result, expected_result)

    def test_identic_columns(self):
        # DO
        result = compute_conversion_rate(202210, self.forecast_data,
                                         self.counter_data, self.input_template_builder)

        # THEN
        expected_result = pd.DataFrame({
            "cell_name": ["A", "B", "C"],
            "traffic_dl_rate": [1.5, 1, 0.5],
            "traffic_ul_rate": [1.5, 1, 0.5],
            "week_period_forecasted": [202210, 202210, 202210]
        })
        columns_result = set(result.columns)
        columns_expected_result = set(expected_result.columns)
        self.assertSetEqual(columns_result, columns_expected_result, "Columns are not identic")


class TestPreProcessLte(unittest.TestCase):

    def _assert_data_frame_equal(self, df_1, df_2, msg):
        """
        Internal Test Case function adding Pandas Data Frame comparison to Test Case
        """
        try:
            pd_testing.assert_frame_equal(df_1, df_2)
        except AssertionError as exception:
            raise self.failureException(msg) from exception

    def setUp(self):
        print(type(data.lte))
        for key, value in data.lte.items():
            print(key)
            print(len(value))
        self.lte = pd.DataFrame(data.lte)
        print(self.lte)
        self.addTypeEqualityFunc(pd.DataFrame, self._assert_data_frame_equal)

    def test_pre_process_lte_headers(self):
        expected_header_lte = pd.DataFrame(data.header_lte)
        _, r_header_lte, _ = pre_process_lte(self.lte)
        self.assertIsInstance(r_header_lte, pd.DataFrame)
        self.assertEqual(expected_header_lte, r_header_lte)

    def test_pre_process_lte_columns(self):
        expected_colmuns_values = data.values_columns_lte
        _, _, r_colnames = pre_process_lte(self.lte)
        self.assertIsInstance(r_colnames.values, object)
        self.assertEqual(len(expected_colmuns_values), len(r_colnames.values))
        self.assertEqual(r_colnames.values[0], 'Site')


class TestPostProcessLte(unittest.TestCase):

    @patch('src.d02_preprocessing.conversion_rate.load_workbook')
    def test_post_process_lte(self, mock_load_workbook):
        mock_workbook = MagicMock()
        mock_sheet = MagicMock()
        mock_sheet.__getitem__.side_effect = lambda x: MagicMock(value='') if x in ('E1', 'E2',
            'E3', 'E4', 'E5', 'E6', 'E7', 'E8') else 'FDD/TDD' if (x == 'E9') else 'Mode'
        mock_workbook.active = mock_sheet
        mock_load_workbook.return_value = mock_workbook
        post_process_lte('dummy_input_path', 'dummy_output_path')
        mock_load_workbook.assert_called_with(filename='dummy_input_path')
        self.assertEqual(mock_sheet['E9'], 'FDD/TDD')
        self.assertEqual(mock_sheet['E10'], 'Mode')
        mock_workbook.save.assert_called_with('dummy_output_path')


if __name__ == '__main__':
    unittest.main()
