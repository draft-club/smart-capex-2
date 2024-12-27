import sys
sys.path.append("C:/Users/CCCB9081/Documents/smart-capex/new")

import os
import unittest
import numpy as np
import geopandas as gpd


from coverage.preprocessing.site_radio import SiteRadio


class TestSiteRadio(unittest.TestCase):

    def setUp(self):
        self.configfile = "C:/Users/CCCB9081/Documents/smart-capex/new/config.ini"
        self.site_radio = SiteRadio()
        config = self.site_radio.get_config(self.configfile)

        self.source_directory = config["directories"]["source_directory"]
        self.data_directory = config["directories"]["data_directory"]
        self.file_site_radio = config["filenames"]["file_site_radio"]
        self.file_dictionary_info = config["filenames"]["file_dictionary_info"]
        self.grid_side_length = float(config["parameters"]["grid_side_length"])
        self.country = "Côte d'Ivoire"

    def test_files_exists(self):
        self.assertTrue(os.path.exists(self.file_site_radio))
        self.assertTrue(os.path.exists(self.file_dictionary_info))

    def test_grid_length(self):
        self.assertIsInstance(self.grid_side_length,float)
        self.assertGreaterEqual(self.grid_side_length, 1)

    def test_entries_in_dict(self):
        dic = self.site_radio.read_dictionary(self.file_dictionary_info)
        self.assertTrue(self.country in dic)
        self.assertGreaterEqual(len(dic[self.country].values()), 5)
        self.assertFalse(np.isnan(list(dic[self.country].values())).any())

    def test_grid_created(self):
        grid_df = self.site_radio.create_grid(self.source_directory, self.data_directory,
                                              self.file_dictionary_info, self.country, self.grid_side_length)
        self.assertIsInstance(grid_df, gpd.geodataframe.GeoDataFrame)
        self.assertTrue(grid_df.shape[0] > 0)


if __name__ == '__main__':
    unittest.main()
