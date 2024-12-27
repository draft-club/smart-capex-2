import unittest
import pandas as pd
from src.d02_preprocessing.OMA.read_process_oss_counter import preprocess_oss_weekly_from_capacity


class TestPreprocessOssWeeklyFromCapacity(unittest.TestCase):

    def setUp(self):
        self.df_oss_weekly_from_capacity = pd.DataFrame({
            'week_period': [202101, 202028, 202052],
            'cell_name': ['BEN001U', 'AGA001_TDD_L1', 'BEN001U'],
            'total_voice_traffic_kerlangs': [0.096567, 0.035471, 0.043477],
            'total_data_traffic_dl_gb': [33.9908, 10.2814, 21.3741],
            'total_data_traffic_ul_gb': [5.6082, 2.2016, 3.2617],
            'average_throughput_dl_kbps': [1554.902429, 2104.550871, 1598.795586],
            'average_power_load_dl': [78.133814, 23.135686, 69.549329],
            'cell_tech': ['3G', '3G', '3G'],
            'traffic_mobile_gb': ['Nan', 'NaN', 'NaN'],
            'traffic_box_gb': ['NaN', 'NaN', 'NaN'],
            'average_active_users': ['NaN', 'NaN', 'NaN'],
            'average_prb_load_dl': ['NaN', 'NaN', 'NaN'],
            'week': [1, 28, 52],
            'year': [2021, 2020, 2020],
            'date': ['2021-01-04', '2020-07-13', '2020-12-28'],
            'month': [1, 7, 12],
            'site_id': ['BEN001', 'BEN001', 'BEN001'],
            'cell_band': ['U2100', 'U2100', 'U2100'],
            'region': ['ORIENTAL', 'ORIENTAL', 'ORIENTAL'],
            'ville': ['BNI ANSAR (MUN.)', 'BNI ANSAR (MUN.)', 'BNI ANSAR (MUN.)'],
            'province': ['NADOR', 'NADOR', 'NADOR']
        })

    def test_preprocess_oss_weekly_from_capacity(self):
        cell_filter = ['TDD']
        use_case = 'TDD'
        result = preprocess_oss_weekly_from_capacity(self.df_oss_weekly_from_capacity,
                                                     cell_filter, use_case)
        self.assertIsInstance(result, pd.DataFrame)

    def test_preprocess_oss_weekly_from_capacity_tdd(self):
        cell_filter = ['TDD']
        use_case = 'TDD'
        result = preprocess_oss_weekly_from_capacity(self.df_oss_weekly_from_capacity,
                                                     cell_filter, use_case)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 21)

    def test_preprocess_oss_weekly_from_capacity_fdd(self):
        cell_filter = ['TDD']
        use_case = 'FDD'
        result = preprocess_oss_weekly_from_capacity(self.df_oss_weekly_from_capacity,
                                                     cell_filter, use_case)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 21)


if __name__ == '__main__':
    unittest.main()
