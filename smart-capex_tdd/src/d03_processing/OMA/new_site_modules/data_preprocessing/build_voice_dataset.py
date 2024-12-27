import os

from src.d00_conf.conf import conf
from src.d03_processing.OMA.new_site_modules.data_preprocessing.build_fdd_dataset import \
 prepare_density_pred_dataset, prepare_density_train_dataset_generic
from src.d03_processing.OMA.new_site_modules.data_preprocessing.data_processing import (
    _compute_deployment_date)
def prepare_density_train_dataset_voice():
    """
    The prepare_density_train_dataset_voice function processes site and traffic data to prepare a
    training dataset for voice traffic density analysis. It involves loading datasets, computing
    deployment dates, identifying neighboring sites, validating groups,
    and calculating traffic features.

    Returns
    -------
    final_dataset : pd.DataFrame
        A cleaned and enriched DataFrame containing the final training dataset with various
        computed features.
    """
    sites_dataset_path = os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES'])
    traffic_dataset_path = (os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'processed_oss_') +
                            conf['USE_CASE'] + '.csv')

    def voice_deployment_date_callback(sites_densif_dataset, traffic_dataset):
        return _compute_deployment_date(sites_densif_dataset, traffic_dataset,
                                        traffic_feature="total_voice_traffic_kerlangs")

    return prepare_density_train_dataset_generic(sites_dataset_path, traffic_dataset_path,
                                                 voice_deployment_date_callback,
                                                 traffic_feature="total_voice_traffic_kerlangs")



def prepare_density_pred_dataset_voice(new_site_dataset_path, fdd_config=None):
    """
    The prepare_density_pred_dataset_voice function prepares a dataset for predicting voice traffic
    density at new site deployments. It integrates data from various sources, computes neighbor
    features, and adds specific bands to the final dataset.

    Parameters
    ----------
    new_site_dataset_path: str
        Path to the new site dataset file
    fdd_config: list
        List of integers representing the FDD configuration, default is [3, 3, 3, 3, 3].

    Returns
    -------
    final_dataset: pd.DataFrame
        A DataFrame containing the final dataset with added bands and computed features for
        predicting voice traffic density.

    """
    return prepare_density_pred_dataset(new_site_dataset_path, fdd_config,
                                        traffic_feature="total_voice_traffic_kerlangs")
