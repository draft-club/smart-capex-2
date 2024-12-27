"""Tests for test_tdd_pipeline.py."""
import unittest
import os
from unittest.mock import patch

from src.tdd_pipeline import prepare_densification_topology_file
from src.d00_conf.conf import conf
from src.tests import data_test


class TestPrepareDensificationTopogyFile(unittest.TestCase):
    """
    Class to test TDD prepare densification topology
    """

    @patch('pandas.DataFrame.to_excel')
    @patch('pandas.read_csv')
    def test_prepare_densification_topology_assert_call(self, mock_read_csv, mock_to_excel):

        mock_read_csv.side_effect = lambda path, sep: data_test.path_to_df[path]
        result = prepare_densification_topology_file()
        print(result)
        mock_to_excel.assert_called_with(os.path.join(conf['PATH']['RANDIM'],
                                                      'topology_randim.xlsx'), index=False)

    @patch('pandas.DataFrame.to_excel')
    @patch('pandas.read_csv')
    def test_prepare_densification_topology_result(self, mock_read_csv, mock_to_excel):
        mock_read_csv.side_effect = lambda path, sep: data_test.path_to_df[path]
        result = prepare_densification_topology_file()
        expected_columns_result = ['Site', 'cell_name', 'cell_id', 'Y_latitude',
                                   'X_longitude', 'azimuth']
        self.assertEqual(len(result.columns), 6)
        self.assertListEqual(list(result.columns), expected_columns_result)
        print(mock_to_excel)
