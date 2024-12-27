import datetime as dt
import unittest

import pandas as pd

from src.d00_conf.conf import conf_loader


conf_loader('fake_country_e2e.json')


@unittest.skip('Not implemented')
class TestTrafficForecasting(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_columns = ["cell_name", "date", "cell_tech", "cell_band", "site_id",
                                 'cell_sector', "year",
                                 "week", "week_period", "average_throughput_user_dl_kbps",
                                 "cell_occupation_dl_percentage", "total_data_traffic_dl_gb"]
        self.dt_begin = dt.datetime(year=1985, month=10, day=25)
        self.date_list = [self.dt_begin + dt.timedelta(days=day) for day in range(2)]
        self.mock_data = pd.DataFrame({
            'cell_name': ['Cell-B-0_L23-000', 'Cell-B-0_L23-000'],
            'date': self.date_list,
            'cell_band': ['L23', 'L23'],
            'cell_tech': ['4G', '4G'],
            'cell_sector': ['0', '0'],
            'year': [1985, 1985],
            'week': [43, 43],
            'week_period': ['198543', '198543'],
            'site_id': ['Cell-B-0', 'Cell-B-0'],
            'total_data_traffic_dl_gb': [0.7, 0.8],
            'average_throughput_user_dl_kbps': [700, 800],
            'average_throughput_user_ul_kbps': [700, 800],
            'cell_occupation_dl_percentage': [70, 80],
            'average_number_of_users_dl': [700, 800],
            'cell_occupation_ul_percentage': [70, 80],
            'total_data_traffic_ul_gb': [0.7, 0.8],
            'average_number_of_users_in_queue': [3, 4]
        })



if __name__ == '__main__':
    unittest.main()
