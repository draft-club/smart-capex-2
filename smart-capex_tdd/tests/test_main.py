import os
import argparse
import unittest
from unittest.mock import patch
import pandas as pd


from main_randim import main
from src.d01_utils.utils import get_last_folder


class TestMain(unittest.TestCase):
    """Class to test main function (end-to-end test)"""
    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(path_to_country_parameters='fake_country_e2e.json'))
    #@patch('configparser.ConfigParser.getboolean', return_value=True)
    def test_main(self, mock_args):
        # Call main munction
        main()

        # Read result
        file_path_result = os.path.join(get_last_folder('data/samples/05_models_output/final_npv'))
        df = pd.read_csv(os.path.join(
            file_path_result,'final_npv_of_the_upgrade_FDD_from_capacity.csv'), sep='|')

        # Check type of result
        self.assertIsInstance(df, pd.DataFrame)

        # Check nb columns
        expected_num_columns = 22
        self.assertEqual(len(df.columns), expected_num_columns)

        # Check name columns
        expected_columns_names = ['NPV', 'total_opex', 'total_revenue',
                                  'EBITDA_Value', 'EBITDA', 'IRR']
        for column_name in expected_columns_names:
            self.assertIn(column_name, df.columns)

        # Check if mocks called
        self.assertTrue(mock_args.called)
        expected_args = ('fake_country_e2e.json',)
        self.assertEqual(mock_args.call_args, expected_args)


if __name__ == '__main__':
    unittest.main()
