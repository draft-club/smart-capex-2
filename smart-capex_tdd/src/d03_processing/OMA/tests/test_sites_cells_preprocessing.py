import unittest
import pandas as pd

from src.d02_preprocessing.OMA.read_process_sites import sites_preprocessing


#from src.d02_preprocessing.OCI.read_process_sites import sites_preprocessing, preprocessing_cells



@unittest.skip('Not implemented')
class TestSitesCellsPreprocessing(unittest.TestCase):
    def test_sites_preprocessing(self):
        df_deployment_plan_sites = sites_preprocessing()

        # Assert that the output is a DataFrame
        self.assertIsInstance(df_deployment_plan_sites, pd.DataFrame)

        # Assert that the columns have the expected names
        expected_columns = ['site_id', 'site_physique', 'site_status', 'site_ville',
                            'site_gestionnaire','site_quartier', 'site_commune', 'site_department',
                            'site_region', 'site_district','site_zone_commerciale',
                            'site_longitude', 'site_latitude', 'site_tower_height','site_type_baie',
                            'site_geotype', 'site_energy_type', 'opex_key']

        self.assertListEqual(list(df_deployment_plan_sites.columns), expected_columns)

        # Assert that the "site_type_baie" column has no null values
        self.assertFalse(df_deployment_plan_sites["site_type_baie"].isnull().any())

        # Assert that the "site_gestionnaire" column is either "IHS" or "ESCO"
        self.assertTrue((df_deployment_plan_sites["site_gestionnaire"] == "IHS").any() or
                        (df_deployment_plan_sites["site_gestionnaire"] == "ESCO").any())

        # Assert that the "site_energy_type" column is either "grid_genset" or "other"
        self.assertTrue((df_deployment_plan_sites["site_energy_type"] == "grid_genset").any() or
                        (df_deployment_plan_sites["site_energy_type"] == "other").any())

        # Assert that the "opex_key" column is in the format "IHS_grid_genset" or "ESCO_Outdoor"
        self.assertTrue((df_deployment_plan_sites["opex_key"].str.startswith("IHS_") &
                         df_deployment_plan_sites["opex_key"].str.endswith("grid_genset")).any() or
                        (df_deployment_plan_sites["opex_key"].str.startswith("ESCO_") &
                         df_deployment_plan_sites["opex_key"].str.endswith("Outdoor")).any())




if __name__ == '__main__':
    unittest.main()
