from src.d03_processing.OMA.new_site_modules.models.traffic_target_baseclass import (
    TrafficTargetSiteModelBase)

class TrafficTargetSiteModelFDD(TrafficTargetSiteModelBase):
    """
    The TrafficTargetSiteModelFDD class is designed to predict traffic at target sites using machine
    learning models. It preprocesses data, trains models using grid search and stratified sampling,
    and evaluates model performance. The class primarily uses an XGBoost regressor within a pipeline
    that includes feature scaling and target transformation.

    """
    def __init__(self, model_type='linear'):
        super().__init__(model_type)
        self.features = ["L1800_estim", "L2600_estim", "L800_estim", "U2100_estim", "U900_estim",
                         "mean_dist",'region_Béni Mellal-Khénifra', 'region_Drâa-Tafilalet',
                         'region_Eddakhla-Oued Eddahab', 'region_Fès-Meknès',
                         'region_Grand Casablanca-Settat', 'region_Guelmim-Oued Noun',
                         'region_Laayoune-Sakia El Hamra', 'region_Marrakech-Safi',
                         'region_Oriental', 'region_Rabat-Salé-Kénitra', 'region_Souss-Massa',
                         'region_Tanger-Tetouan-Al Hoceima', 'cell_tech_3G', 'cell_tech_4G',
                         'L1800_target', 'L2600_target', 'L800_target', 'U2100_target',
                         'U900_target', 'L1800_neighbor', 'L1800_traffic', 'L2600_neighbor',
                         'L2600_traffic', 'L800_neighbor', 'L800_traffic', 'U2100_neighbor',
                         'U2100_traffic', 'U900_neighbor', 'U900_traffic', "neighbor_traffic"]


    def _compute_estim(self, dataset):
        """
        The _compute_estim method calculates estimated traffic values for different frequency bands
        by normalizing the traffic of each band with its neighboring traffic and then scaling it by
        a target value. This helps in creating new features that can be used for further modeling.

        Parameters
        ----------
        dataset: pd.DataFrame
            A pandas DataFrame containing traffic, neighbor traffic, and target values for different
            frequency bands.

        Returns
        -------
        dataset: pd.DataFrame
            The method returns the input dataset with additional columns for the estimated traffic
            values (L1800_estim, L2600_estim, L800_estim, U2100_estim, U900_estim)

        """
        dataset["L1800_estim"] = ((dataset["L1800_traffic"] / (dataset["L1800_neighbor"] + 1e-8)) *
                                  dataset["L1800_target"])
        dataset["L2600_estim"] = ((dataset["L2600_traffic"] / (dataset["L2600_neighbor"] + 1e-8)) *
                                  dataset["L2600_target"])
        dataset["L800_estim"] = ((dataset["L800_traffic"] / (dataset["L800_neighbor"] + 1e-8)) *
                                 dataset["L800_target"])
        dataset["U2100_estim"] = ((dataset["U2100_traffic"] / (dataset["U2100_neighbor"] + 1e-8)) *
                                  dataset["U2100_target"])
        dataset["U900_estim"] = ((dataset["U900_traffic"] / (dataset["U900_neighbor"] + 1e-8)) *
                                 dataset["U900_target"])
        return dataset
