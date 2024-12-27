import unittest

import pandas as pd

from src.d03_processing.OMA.tests.data_output_to_bd import data_df_predicted, data_site
from src.d03_processing.OMA.output_to_bd import prepare_forecast_to_db, getjson_feature, \
    getband_or_tech


class TestPrepareForecastToDB(unittest.TestCase):
    def setUp(self):
        self.df_predicted_traffic_kpis = pd.DataFrame(data_df_predicted)
        self.site = pd.DataFrame(data_site)

    def test_type_result_prepare_forecast_to_bd(self):
        df_predicted_traffic_kpis = prepare_forecast_to_db(self.df_predicted_traffic_kpis,
                                                           self.site)
        self.assertIsInstance(df_predicted_traffic_kpis, pd.DataFrame)

    def test_columns_prepare_forecast_to_bd(self):
        df_predicted_traffic_kpis = prepare_forecast_to_db(self.df_predicted_traffic_kpis,
                                                           self.site)
        self.assertEqual(df_predicted_traffic_kpis.shape[1], 19)


class TestGetJsonFeature(unittest.TestCase):
    def setUp(self):
        self.df_predicted_traffic_kpis = pd.DataFrame(data_df_predicted)
        self.period = "week_period"
        self.key = "site_id"

    def test_type_result_get_json_feature(self):
        result = getjson_feature(self.df_predicted_traffic_kpis, self.period, self.key)
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_get_json_feature(self):
        result = getjson_feature(self.df_predicted_traffic_kpis, self.period, self.key)
        self.assertEqual(result.shape[1], 5)

    def test_columns_name_get_json_feature(self):
        result = getjson_feature(self.df_predicted_traffic_kpis, self.period, self.key)
        self.assertIn('period', result.columns)
        self.assertIn('type_period', result.columns)


class TestGetBandOrTechnical(unittest.TestCase):
    def setUp(self):
        self.df_predicted_traffic_kpis = pd.DataFrame(data_df_predicted)
        keys = ["site_id"]
        # list of periods
        periods = ["week_period", "year"]
        by = "cell_tech"
        # list of periods
        for _, key in enumerate(keys):
            for period in periods:
                feature_results = getjson_feature(self.df_predicted_traffic_kpis,
                                                  key=key, period=period)
                feature_results.period = feature_results.period.astype(str)
                by = "cell_tech"
                self.config = {'groupby_cols': [key, period, by],
                          'value': "total_data_traffic_dl_gb",
                          'key': key,
                          'period': period}

    def test_type_result_getband_or_tech(self):
        result_type = getband_or_tech(self.df_predicted_traffic_kpis,self.config)
        self.assertIsInstance(result_type, pd.DataFrame)

    def test_columns_getband_or_tech(self):
        result_nb_colmuns = getband_or_tech(self.df_predicted_traffic_kpis,self.config)
        self.assertEqual(result_nb_colmuns.shape[1], 3)

    def test_columns_name_getband_or_tech(self):
        result_name_columns = getband_or_tech(self.df_predicted_traffic_kpis,self.config)
        self.assertIn('period', result_name_columns.columns)
        self.assertIn('id', result_name_columns.columns)

if __name__ == '__main__':
    unittest.main()
