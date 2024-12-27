import os

from src.d03_processing.OMA.new_site_modules.data_preprocessing.data_source import (
    get_sites_dataset,get_densif_sites_dataset, get_traffic_dataset, get_randim_output_dataset)
from src.d03_processing.OMA.new_site_modules.data_preprocessing.data_processing import (
    _compute_deployment_date, _compute_neighbors, _make_traffic_dataset, _check_valid_groups,
    _prepare_final_dataset_train,_compute_neighbor_features, _compute_target_site_bands,
    _compute_neighbor_bands, _prepare_final_dataset_pred, __add_bands)
from src.d00_conf.conf import conf


def prepare_density_train_dataset_generic(sites_dataset_path, traffic_dataset_path,
                                          deployment_date_callback, traffic_feature=None):
    """
    The prepare_density_train_dataset_generic function prepares a training dataset for a
    density-based model by processing site and traffic data. It integrates various preprocessing
    steps, including loading datasets, computing deployment dates, identifying neighboring sites,
    and calculating traffic features. The final dataset is cleaned and enriched with neighbor and
    band features before being returned.

    Parameters
    ----------
    sites_dataset_path: str
        Path to the sites dataset file.
    traffic_dataset_path: str
        Path to the traffic dataset file.
    deployment_date_callback: function
        Callback function to compute deployment date.
    traffic_feature: str, optional
        Specific traffic feature to use for dataset preparation, default is None.

    Returns
    -------
    final_dataset: pd.DataFrame
        A cleaned and enriched DataFrame containing the final training dataset with various
        computed features.
    """
    sites_dataset = get_sites_dataset(sites_dataset_path)
    sites_densif_dataset = get_densif_sites_dataset(sites_dataset)
    traffic_dataset = get_traffic_dataset(traffic_dataset_path)

    deployment_localization = deployment_date_callback(sites_densif_dataset, traffic_dataset)

    group_dataset, dico_group = _compute_neighbors(deployment_localization, sites_dataset,
                                                   5, 7)
    group_traffic_dataset = _make_traffic_dataset(group_dataset, traffic_dataset,
                                                  20, 8,
                                                  traffic_feature=traffic_feature)

    # Tmp save
    group_traffic_dataset.to_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                    f'group_traffic_dataset_{traffic_feature if traffic_feature else "fdd"}.csv'),
                    index=False, sep='|')

    group_traffic_dataset = _check_valid_groups(group_traffic_dataset, dico_group)
    final_dataset = _prepare_final_dataset_train(group_traffic_dataset)
    neighbor_features = _compute_neighbor_features(group_traffic_dataset)
    bands_target = _compute_target_site_bands(group_traffic_dataset, traffic_dataset)
    bands_neighbors = _compute_neighbor_bands(group_traffic_dataset, traffic_dataset)
    print("end features")

    final_dataset = final_dataset.join(neighbor_features)
    final_dataset = final_dataset.join(bands_target)
    final_dataset = final_dataset.join(bands_neighbors)
    final_dataset = final_dataset.dropna().drop_duplicates()

    return final_dataset


def prepare_density_train_dataset_fdd():
    """
    The prepare_density_train_dataset_fdd function prepares a training dataset for a density-based
    model by processing site and traffic data. It integrates various preprocessing steps, including
    loading datasets, computing deployment dates, identifying neighboring sites, and calculating
    traffic features. The final dataset is cleaned and enriched with neighbor and band features
    before being returned.

    Returns
    -------
    final_dataset: pd.DataFrame
        A cleaned and enriched DataFrame containing the final training dataset with various
        computed features.
    """
    sites_dataset_path = os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES'])
    traffic_dataset_path = os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'processed_oss_') + conf[
        'USE_CASE'] + '.csv'

    return prepare_density_train_dataset_generic(sites_dataset_path, traffic_dataset_path,
                                                 _compute_deployment_date)


def prepare_density_pred_dataset_fdd(new_site_dataset_path, fdd_config=None):
    """
    The prepare_density_pred_dataset_fdd function prepares a dataset for predicting traffic density
    at new sites. It processes site and traffic data, computes neighbor features, and adds necessary
    bands to the dataset.

    Parameters
    ----------
    new_site_dataset_path: str
        Path to the new site dataset file.
    fdd_config: list
        Configuration list for FDD bands.

    Returns
    -------
    final_dataset : pd.DataFrame
        A DataFrame containing the prepared dataset with neighbor features and bands for traffic
        density prediction.
    """
    return prepare_density_pred_dataset(new_site_dataset_path, fdd_config)



def pre_process_build_dataset(fdd_config, new_site_dataset_path):
    if fdd_config is None:
        fdd_config = [3, 3, 3, 3]
    deployment_localization = get_randim_output_dataset(new_site_dataset_path)
    sites_dataset = get_sites_dataset(
        os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']))
    # traffic_dataset = get_traffic_dataset("data/traffic_weekly_predicted_kpis_FDD.csv")
    traffic_dataset = get_traffic_dataset(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'traffic_weekly_predicted_kpis_') + conf[
            'USE_CASE'] + '.csv')
    group_dataset, _ = _compute_neighbors(deployment_localization, sites_dataset,
                                          5, 2)
    return fdd_config, group_dataset, traffic_dataset



def prepare_density_pred_dataset(new_site_dataset_path, fdd_config, traffic_feature=None):
    """
    The prepare_density_pred_dataset function prepares a dataset for predicting traffic density
    at new sites. It processes site and traffic data, computes neighbor features, and adds necessary
    bands to the dataset.

    Parameters
    ----------
    new_site_dataset_path: str
        Path to the new site dataset file.
    fdd_config: list
        Configuration list for FDD bands.
    traffic_feature: str, optional
        Specific traffic feature to use for dataset preparation, default is None.

    Returns
    -------
    final_dataset : pd.DataFrame
        A DataFrame containing the prepared dataset with neighbor features and bands for traffic
        density prediction.
    """
    fdd_config, group_dataset, traffic_dataset = pre_process_build_dataset(fdd_config,
                                                                           new_site_dataset_path)
    group_traffic_dataset = _make_traffic_dataset(
        group_dataset[group_dataset.distance_from_deployment != 0],
        traffic_dataset, -1, 8, traffic_feature=traffic_feature)

    final_dataset = _prepare_final_dataset_pred(group_dataset)
    neighbor_features = _compute_neighbor_features(group_traffic_dataset)
    bands_neighbors = _compute_neighbor_bands(group_traffic_dataset, traffic_dataset)
    print("end features")

    final_dataset = final_dataset.join(neighbor_features)
    final_dataset = final_dataset.join(bands_neighbors)
    final_dataset = final_dataset.dropna().drop_duplicates()
    final_dataset = __add_bands(final_dataset, fdd_config)

    return final_dataset
