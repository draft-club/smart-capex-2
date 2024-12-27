from src.d03_processing.OMA.new_site_modules.models.traffic_target_baseclass import (
    TrafficTargetSiteModelBase)

class TrafficTargetSiteModelVoice(TrafficTargetSiteModelBase):
    """
    The TrafficTargetSiteModelVoice class is designed to model and predict traffic at target sites
    using machine learning techniques. It primarily utilizes the XGBoost regressor within a pipeline
    that includes feature scaling and target transformation. The class provides methods for training
    the model with grid search for hyperparameter tuning, training with a specific dataset,
    and making predictions.

    """
    def __init__(self, model_type='linear'):
        super().__init__(model_type)
        self.features = ["neighbor_traffic", 'mean_dist', "traffic_3g", 'cell_tech_3G',
                         'cell_tech_4G', "nb_sites"]

    def _compute_estim(self, dataset):
        return dataset[(dataset.target_traffic <= 2)].dropna()
