"""Apply train gain densification"""
import logging
import os
from datetime import datetime as dt
from math import sin, cos, sqrt, atan2, radians

import haversine as hs
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import spatial

from src.d00_conf.conf import conf
from src.d01_utils.utils import add_logging_info
from src.d03_processing.OMA.technical_modules.train_densification_impact_model import \
    cleaning_new_sites

EARTH_RADIUS = 6373.0



def compute_distance_between_sites(site):
    """
    Compute the distance between site's dataset

    Parameters
    ----------
    site: pd.Dataframe
        Dataset with site's information

    Returns
    -------
    distance: pd.DataFrame
        Matrix with all distance between sites
    """
    site = site[['longitude', 'latitude', 'site_id']].drop_duplicates(subset='site_id')
    all_points = site[['latitude', 'longitude']].values
    distance_matrix = spatial.distance.cdist(all_points,
                                             all_points,
                                             hs.haversine)
    distance = pd.DataFrame(distance_matrix, index=site['site_id'].values,
                            columns=site['site_id'].values)
    return distance


def compute_kpis_before_after(new_sites):
    """
    The compute_kpis_before_after function processes a dataset of new sites to determine key
    performance indicators (KPIs) before and after an upgrade.
    It classifies the data based on the upgrade date, computes the KPIs for both periods,
    and returns the processed datasets

    Parameters
    ----------
    new_sites: pd.DataFrame

    Returns
    -------
    new_sites: pd.DataFrame
        New site's dataset
    kpi_before : pd.DataFrame
        Dataset with kpi before upgrade
    kpi_after: pd.DataFrame
        Dataset with kpi after upgrade
    """
    # Compute kpi feature (check with deployment date if it's before or after)
    compute_kpi_features(new_sites)
    kpi_before = compute_kpi_before(new_sites)
    kpi_after = compute_kpi_after(new_sites)
    return new_sites, kpi_before, kpi_after


def compute_kpi_features(new_sites):
    """
    The compute_kpi_features function classifies each row in the new_sites DataFrame based on the
    date of an upgrade. It assigns a label indicating whether the date is before, after,
    or unrelated to the upgrade period.

    Parameters
    ----------
    new_sites : pd.DataFrame
        A pandas DataFrame containing a 'date' column.

    Returns
    -------
    new_sites : pd.DataFrame
        The input DataFrame with an additional 'kpi_features' column indicating the classification.
    """
    week_of_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
    d = week_of_upgrade[:-2] + '-W' + week_of_upgrade[4:]
    date_week_upgrade = dt.strptime(d + '-1', "%Y-W%W-%w")
    new_sites['kpi_features'] = new_sites.apply(lambda x:
                                                np.where(
                                                    (((x.date > date_week_upgrade + relativedelta(
                                                        weeks=-9)) and
                                                      (x.date < date_week_upgrade))),
                                                    'kpi_before',
                                                    np.where(
                                                        ((
                                                        x.date > date_week_upgrade + relativedelta(
                                                                 weeks=3))
                                                         and (
                                                        x.date < date_week_upgrade + relativedelta(
                                                                 weeks=12))),
                                                        'kpi_after', 'nothing')
                                                ), axis=1)


def compute_kpi_after(new_sites):
    """
    The compute_kpi_after function filters and processes a dataset to compute key performance
    indicators (KPIs) for sites after an upgrade. It groups the data by site ID and calculates the
    mean values for traffic, voice, and average PRB, then renames the columns and fills any missing
    values with zero.

    Parameters
    ----------
    new_sites : pd.DataFrame
        A pandas DataFrame containing site data with columns for site ID, KPI features, traffic,
        voice, and average PRB.

    Returns
    -------
    kpi_after : pd.DataFrame
        A pandas DataFrame with columns for site ID, mean traffic, mean voice, and mean average PRB
        after the upgrade, with missing values filled with zero

    """
    kpi_after = new_sites[new_sites['kpi_features'] == 'kpi_after']
    kpi_after = kpi_after.groupby('site_id').agg(
        {'n_traffic': np.nanmean, 'n_voice': np.nanmean,'n_avgprb': np.nanmean}).reset_index()
    kpi_after.columns = ['site_id', 'n_traffic_after', 'n_voice_after', 'n_avgprb_after']
    kpi_after.fillna(0, inplace=True)
    return kpi_after


def compute_kpi_before(new_sites):
    """
    The compute_kpi_before function filters and processes a dataset to compute key performance
    indicators (KPIs) for sites before an upgrade. It groups the data by site_id and calculates
    the mean of several metrics

    Parameters
    ----------
    new_sites : pd.DataFrame
         A pandas DataFrame containing site data with columns such as site_id, kpi_features,
         n_traffic, n_voice, n_avgthrput, n_avgprb, and n_avgusr.

    Returns
    -------
    kpi_before : pd.DataFrame
        A pandas DataFrame with columns site_id, n_traffic_before, n_voice_before, n_avgprb_before,
        n_avgthrput_before, and n_avgusr_before,
        containing the computed KPIs for each site before the upgrade.

    """
    kpi_before = new_sites[new_sites.kpi_features == 'kpi_before']
    kpi_before = kpi_before.groupby('site_id').agg(
        {'n_traffic': np.nanmean, 'n_voice': np.nanmean, 'n_avgthrput': np.nanmean,
         'n_avgprb': np.nanmean, 'n_avgusr': np.nanmean}).reset_index()
    kpi_before.columns = ['site_id', 'n_traffic_before', 'n_voice_before',
                          'n_avgprb_before', 'n_avgthrput_before', 'n_avgusr_before']
    return kpi_before


def prepare_improvement_model(kpis, distance_randim_site):
    """
    Function to prepare the improvement model for comparison

    Parameters
    ----------
    kpis: pd.DataFrame
        Dataset with Kpis
    distance_randim_site: pd.DataFrame

    Returns
    -------
    new_sites: pd.DataFrame
    """
    # Add historical traffic by site
    kpis_site = kpis.groupby(['site_id', 'date']).agg({'total_data_traffic_dl_gb': 'sum',
                                                       'total_voice_traffic_kerlangs': 'sum',
                                                       'average_prb_load_dl': 'sum',
                                                       'average_throughput_dl_kbps': 'sum',
                                                       'average_active_users': 'sum'}).reset_index()
    kpis_site['date'] = kpis_site.date.apply(lambda x: dt.strptime(x, "%Y-%m-%d"))

    # Traffic for neighbour_1
    new_sites = distance_randim_site[['site_id', 'neighbour_1', 'neighbour_2',
                                      'commune', 'ville', 'province', 'region']] \
        .merge(kpis_site, left_on='neighbour_1', right_on='site_id', how='left')
    new_sites = new_sites.merge(kpis_site, left_on='neighbour_2', right_on='site_id', how='right',
                                suffixes=['_n1', '_n2'])
    new_sites = cleaning_new_sites(new_sites)
    distance_randim_site_just_n2 = distance_randim_site[
        pd.isna(distance_randim_site['neighbour_2'])]
    sites_just_n1 = distance_randim_site_just_n2[['site_id', 'neighbour_1', 'neighbour_2',
                                                  'commune', 'ville', 'province', 'region']] \
        .merge(kpis_site, left_on='neighbour_1', right_on='site_id', how='left')
    sites_just_n1.drop('site_id_y', axis=1, inplace=True)
    sites_just_n1.columns = ['site_id_x', 'neighbour_1', 'neighbour_2', 'commune', 'ville',
                             'province', 'region', 'date_n1',
                             'total_data_traffic_dl_gb_n1', 'total_voice_traffic_kerlangs_n1',
                             'average_prb_load_dl_n1',
                             'average_throughput_dl_kbps_n1', 'average_active_users_n1']

    sites_just_n1['total_data_traffic_dl_gb_n2'] = np.nan
    sites_just_n1['total_voice_traffic_kerlangs_n2'] = np.nan
    sites_just_n1['average_prb_load_dl_n2'] = np.nan
    sites_just_n1['average_throughput_dl_kbps_n2'] = np.nan
    sites_just_n1['average_active_users_n2'] = np.nan

    new_sites = pd.concat([new_sites, sites_just_n1])
    new_sites.columns = ['site_id', 'neighbour_1', 'neighbour_2',
                         'commune', 'ville', 'province', 'region', 'date',
                         'n1_traffic', 'n1_voice', 'n1_avgprb', 'n1_avgusr', 'n1_avgthrput',
                         'n2_traffic', 'n2_voice', 'n2_avgprb', 'n2_avgusr', 'n2_avgthrput']

    new_sites['n_traffic'] = new_sites.apply(lambda x: np.nanmean([x.n1_traffic, x.n2_traffic]),
                                             axis=1)
    new_sites['n_voice'] = new_sites.apply(lambda x: np.nanmean([x.n1_voice, x.n2_voice]),
                                             axis=1)
    new_sites['n_avgprb'] = new_sites.apply(lambda x: np.nanmean([x.n1_avgprb, x.n2_avgprb]),
                                            axis=1)
    new_sites['n_avgusr'] = new_sites.apply(lambda x: np.nanmean([x.n1_avgusr, x.n2_avgusr]),
                                            axis=1)
    new_sites['n_avgthrput'] = new_sites.apply(
        lambda x: np.nanmean([x.n1_avgthrput, x.n2_avgthrput]), axis=1)
    new_sites = new_sites[['site_id', 'neighbour_1', 'neighbour_2',
                           'commune', 'ville', 'province', 'region', 'date',
                           'n_traffic', 'n_voice', 'n_avgprb', 'n_avgusr', 'n_avgthrput']]
    new_sites = new_sites.sort_values(['site_id', 'date'], ascending=True)
    new_sites = new_sites.drop_duplicates()
    return new_sites


def calcul_spatial_distance_in_dataframe(row):
    """
    Fonction to calcul distance between 2 site
    We need columns latitude and longitude for each site

    Parameters
    ----------
    row: Row of panda datafrmae

    Return
    ------
    distance: float
        Distance between 2 site in kilometers
    """
    # Get every value for the compute
    lat1 = row['Location Latitude']
    long1 = row['Location Longitude']
    lat2 = row['latitude_congest_cells']
    long2 = row['longitude_congest_cells']

    # convert decimal degrees to radians
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

    # Compute Distance
    d_lon = long2 - long1
    d_lat = lat2 - lat1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = EARTH_RADIUS * c
    return distance


def prepare_randim_file(randim_file, site):
    """
    Function who take randim file and add site_id for each congest cells
    And ad the distance between the site and the new site

    Parameters
    ----------
    randim_file : pd.DataFrame
        Data from randim file
    site: pd.DataFrame

    Returns
    -------
    cln: pd.DataFrame

    nb_cell_4g: pd.DataFrame
    """
    # Delete headers of randim file
    randim_file = randim_file.iloc[1:, :]

    # Create new columns with list of congested cells
    randim_file['congested_cells'] = randim_file["Congested Cells 4G"].apply(lambda x: x.split(','))
    randim_file_exploded = randim_file.explode('congested_cells')

    # Add site_id for each congest cells
    randim_file_exploded_merge = randim_file_exploded.merge(
        site[['site_id', 'latitude', 'longitude', 'cell_name',
              'commune', 'ville', 'province', 'region']],
        left_on='congested_cells', right_on='cell_name', how='left')
    # Rename the new latitude and longitude
    randim_file_exploded_merge = randim_file_exploded_merge.rename(
        columns={'latitude': 'latitude_congest_cells',
                 'longitude': 'longitude_congest_cells'})
    # Drop first NaN columns
    randim_file_exploded_merge.drop(columns=randim_file_exploded_merge.columns[0], axis=1,
                                    inplace=True)

    # Apply function calcul_spatial_distance_in_dataframe on each row
    randim_file_exploded_merge['distance'] = randim_file_exploded_merge.apply(
        calcul_spatial_distance_in_dataframe, axis=1)
    # Attribute a rank for distance between site
    randim_file_exploded_merge['rank'] = randim_file_exploded_merge.groupby('Cells')[
        'distance'].rank(method='dense')
    randim_file_exploded_merge['rank'].fillna(0, inplace=True)
    randim_file_exploded_merge['rank'] = randim_file_exploded_merge['rank'].astype(int)

    # Keep only 2 neighbors
    cln = randim_file_exploded_merge.loc[randim_file_exploded_merge['rank'] < 3]

    cln = cln.groupby(['Cells', 'site_id'])['rank'].unique().reset_index()
    cln['rank'] = cln['rank'].astype(int)
    cln = cln.pivot(index='Cells', columns='rank', values='site_id')
    cln = cln.reset_index()
    cln = cln.rename(columns={'Cells': 'site_id', 1: 'neighbour_1', 2: 'neighbour_2'})
    cln = cln.merge(site[['site_id', 'commune', 'ville', 'province', 'region']],
                    left_on='neighbour_1', right_on='site_id', how='left')
    cln = cln.rename(columns={'site_id_x': 'site_id'})

    # drop duplicates
    cln = cln.drop_duplicates()
    cln = cln.drop(['site_id_y'], axis=1)
    nb_cell_4g = count_neighbour_site(site, cln)
    return cln, nb_cell_4g


def count_neighbour_site(site, cln, tech='4G'):
    """
    Function use to count cell name with specific tech for each neighbour

    Params
    ------
    site: pd.DataFrame
        Dataset with site's information
    cln: pd.DataFrame
        Dataset with the new site creation and these two neighbors

    Return
    ------
    nb_cell_4g: pd.DataFrame
        Dataset with the new site creation these two neighbors and with cell number
    """
    site = site[['site_id', 'cell_name', 'cell_tech']]

    # Filter on 4G cell
    site = site[site.cell_tech == tech]

    # Merge cln with site4g to get cell name for each neighbour
    cln_neighbour_1 = cln[['site_id', 'neighbour_1']]
    cln_neighbour_2 = cln[['site_id', 'neighbour_2']]
    nb_cell_4g_1 = cln_neighbour_1.merge(site, left_on='neighbour_1', right_on='site_id')
    nb_cell_4g_2 = cln_neighbour_2.merge(site, left_on='neighbour_2', right_on='site_id')

    # group by site_id
    nb_cell_4g_groupby_1 = nb_cell_4g_1.groupby('neighbour_1').agg(
        {'cell_name': 'count'}).reset_index()
    nb_cell_4g_groupby_2 = nb_cell_4g_2.groupby('neighbour_2').agg(
        {'cell_name': 'count'}).reset_index()

    # Regroup each neighbour in one dataframe
    nb_cell_4g = cln.merge(nb_cell_4g_groupby_1, how='left', on='neighbour_1')
    nb_cell_4g = nb_cell_4g.merge(nb_cell_4g_groupby_2, how='left', on='neighbour_2')

    # Refactor and cleaning
    nb_cell_4g = nb_cell_4g.rename(columns={'cell_name_x': 'nb_cell_4G_neighbour_1',
                                            'cell_name_y': 'nb_cell_4G_neighbour_2'})
    nb_cell_4g['nb_cell_4G_neighbour_2'] = nb_cell_4g['nb_cell_4G_neighbour_2'].fillna(0)
    nb_cell_4g['nb_cell_4G'] = nb_cell_4g['nb_cell_4G_neighbour_1'] + nb_cell_4g[
        'nb_cell_4G_neighbour_2']
    nb_cell_4g = nb_cell_4g[['site_id', 'neighbour_1', 'neighbour_2', 'nb_cell_4G']]
    nb_cell_4g['nb_cell_4G'] = nb_cell_4g['nb_cell_4G'].fillna(0).astype(int)

    return nb_cell_4g


def create_site_to_model(pred, randim_file, site):
    """
    Function to create from site file to model to apply the model
    Parameters
    ----------
    pred: pd.DataFrame
        Dataset with predictions
    randim_file: pd.DataFrame
        Dataset with randim information
    site: pd.DataFrame
        Dataset with site's information

    Returns
    -------
    site_to_model: pd.DataFrame
    kpi_after: pd.DataFrame
    nb_cell_4g: pd.DataFrame
    """

    # Function who take randim file and add site_id for each congest cells
    #  And ad the distance between the site and the new site
    cln, nb_cell_4g = prepare_randim_file(randim_file, site)

    # Work on Congest cells format
    randim_file = randim_file.iloc[1:, :]
    randim_file['congested_cells'] = randim_file["Congested Cells 4G"].apply(lambda x: x.split(','))
    randim_file_exploded = randim_file.explode('congested_cells')
    l = []
    for cell in randim_file_exploded.congested_cells:
        l.append(cell)
    #pred_randim_cell = pred[pred.cell_name.isin(l)]

    distance_randim_site = cln

    #
    new_sites = prepare_improvement_model(pred, distance_randim_site)
    new_sites_before, kpi_before, kpi_after = compute_kpis_before_after(new_sites)
    site_to_model = prepare_site_to_model(kpi_before, new_sites_before)
    return site_to_model, kpi_after, nb_cell_4g


def prepare_site_to_model(kpi_before, new_sites_before):
    """
    The prepare_site_to_model function merges KPI data with site information,
    selects specific columns, and removes duplicate entries to prepare a dataset for modeling
    .
    Parameters
    ----------
    kpi_before : pd.DataFrame
         DataFrame containing KPI data before improvements
    new_sites_before : pd.DataFrame
        DataFrame containing information about new sites before improvements.

    Returns
    -------
    site_to_model : pd.DataFrame
        DataFrame containing merged KPI and site information, with duplicates removed.
    """
    columns_to_select = ['site_id', 'region', 'ville', 'commune', 'province']
    site_to_model = new_sites_before[columns_to_select]
    site_to_model = kpi_before.merge(site_to_model, on='site_id', how='left')
    site_to_model = site_to_model.drop_duplicates()
    return site_to_model


def predict_impact_kpis(site_to_model):
    """
    Function to predict impact of kpis

    Parameters
    ----------
    site_to_model: pd.DataFrame

    Returns
    -------
    prediction_final: pd.DataFrame
        Final prediction Dataset
    """
    rf_neighbour_traf = joblib.load(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_traffic_neighbour.joblib'))
    rf_neighbour_voice = joblib.load(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_voice_neighbour.joblib'))
    rf_neighbour_prb = joblib.load(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_prb_neighbour.joblib'))
    site_id = site_to_model[['site_id']]
    to_pred = site_to_model.drop('site_id', axis=1)
    col_to_model = ['n_traffic_before', 'n_avgthrput_before', 'n_avgprb_before','n_avgusr_before']
    to_pred_ = to_pred[col_to_model]
    predictions_traffic = rf_neighbour_traf.predict(to_pred_)
    predictions_traffic = pd.DataFrame(predictions_traffic, columns=['pred_traffic_data'])
    predictions_prb = rf_neighbour_prb.predict(to_pred_)
    predictions_prb = pd.DataFrame(predictions_prb, columns=['pred_prb'])
    col_to_model = ['n_avgthrput_before', 'n_avgprb_before',
                    'n_avgusr_before', 'n_voice_before']
    to_pred_voice = to_pred[col_to_model]
    predictions_voice = rf_neighbour_voice.predict(to_pred_voice)
    predictions_voice = pd.DataFrame(predictions_voice, columns=['pred_voice'])


    prediction_final = pd.concat([site_id, predictions_traffic,
                                  predictions_prb, predictions_voice], axis=1)
    return prediction_final


def compare_prediction_after(prediction_model, kpi_after):
    """
    Function to compare the prediction after model

    Parameters
    ----------
    prediction_model: pd.DataFrame
    kpi_after: pd.DataFrame

    Returns
    -------
    to_compare: pd.DataFrame
    """
    to_compare = prediction_model.merge(kpi_after, on='site_id', how='left')
    to_compare['diff_traffic'] = abs(to_compare.n_traffic_after - to_compare.pred_traffic)
    to_compare['diff_prb'] = abs(to_compare.n_avgprb_after - to_compare.pred_prb)
    return to_compare


def scale_encode_features(pred, site):
    """
    Function to encode features based on scaler create during training

    Parameters
    ----------
    pred: pd.DataFrame
    site: pd.DataFrame

    Returns
    -------
    pred: pd.DataFrame
    site: pd.DataFrame
    """
    le_commune = joblib.load(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_commune.joblib'))
    le_province = joblib.load(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_province.joblib'))
    le_region = joblib.load(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_region.joblib'))
    le_ville = joblib.load(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_ville.joblib'))
    site.region = le_region.transform(site.region)
    site['ville'] = le_ville.transform(site['ville'])
    site.commune = le_commune.transform(site.commune)
    site.province = le_province.transform(site.province)
    return pred, site


@add_logging_info
def pipeline_apply_traffic_gain_densification(randim_file, prediction_file, site_file):
    """
    Function to apply the traffic gain densification

    Parameters
    ----------
    randim_file: pd.DataFrame
        Dataset with randim's information
    prediction_file: pd.DataFrame
        Dataset with prediction's information
    site_file: pd.DataFrame
        Dataset with site's information
    tech: str
        Can be '4G' or other

    Returns
    -------
    site_to_model: pd.DataFrame
    prediction: pd.DataFrame
    to_compare: pd.DataFrame
    """
    # Function to encode features based on scaler create during training
    prediction_file, site_file = scale_encode_features(prediction_file, site_file)

    # Function to create from site file to model to apply the model
    site_to_model, kpi_after, nb_cell_4g = create_site_to_model(prediction_file, randim_file,
                                                                site_file)
    logging.info(kpi_after.shape)
    # Merge site_to_model with nb_cell 4g
    site_to_model = site_to_model.merge(nb_cell_4g, on='site_id', how='left')

    # Apply model to the data
    prediction = predict_impact_kpis(site_to_model)
    to_compare = None
    return site_to_model, prediction, to_compare


def compute_traffic_improvement(site_to_model, prediction, kpi='data', site=True):
    """
    The compute_traffic_improvement function calculates the predicted traffic improvement
    for a given site based on densification upgrades. It processes the input data, merges it with
    predictions, and computes the traffic improvement either directly or as a difference from the
    baseline traffic

    Parameters
    ----------
    site_to_model : pd.DataFrame
        DataFrame containing site IDs and baseline traffic data.
    prediction : pd.DataFrame
        DataFrame containing site IDs and predicted traffic data
    kpi : str
        String indicating the key performance indicator ('data' or 'voice').
    site : bool
        Boolean indicating whether to use the predicted traffic directly or compute the improvement.

    Returns
    -------
    df_predicted_increase_in_traffic_by_densifcation_merge : pd.DataFrame
        DataFrame containing the predicted traffic improvement for each site,
        along with additional metadata.
    """

    # Preprocessing Dataset
    df_predicted_increase_in_traffic_by_densifcation = pd.DataFrame(
        prediction['site_id'].drop_duplicates(), columns=['site_id'])
    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
    df_predicted_increase_in_traffic_by_densifcation["week_of_the_upgrade"] = week_of_the_upgrade
    df_predicted_increase_in_traffic_by_densifcation["bands_upgraded"] = "densification"

    # Filter on column kpi
    if kpi == 'data':
        column_to_compute = 'n_traffic_before'
        column_inprovement = 'trafic_improvement'
    else:
        column_to_compute = 'n_voice_before'
        column_inprovement = 'trafic_improvement_voice'
    df_predicted_increase_in_traffic_by_densifcation = (
        df_predicted_increase_in_traffic_by_densifcation.merge(
            site_to_model[['site_id', column_to_compute]].drop_duplicates(),
            how="inner", on="site_id"))


    df_predicted_increase_in_traffic_by_densifcation = (
        df_predicted_increase_in_traffic_by_densifcation.merge(
            prediction, how="inner", on="site_id"))

    # Rename columns prediction
    if kpi == 'data':
        df_predicted_increase_in_traffic_by_densifcation = (
            df_predicted_increase_in_traffic_by_densifcation.rename(columns={"pred_traffic_data":
                                                                             "predicted_traffic"}))
    else:
        df_predicted_increase_in_traffic_by_densifcation = (
            df_predicted_increase_in_traffic_by_densifcation.rename(columns={"pred_voice":
                                                                             "predicted_traffic"}))

    if site:
        df_predicted_increase_in_traffic_by_densifcation[column_inprovement] = (
            df_predicted_increase_in_traffic_by_densifcation.predicted_traffic)
    else:

        df_predicted_increase_in_traffic_by_densifcation[
            column_inprovement] = (
                    df_predicted_increase_in_traffic_by_densifcation.predicted_traffic -
                    df_predicted_increase_in_traffic_by_densifcation[column_to_compute])
        df_predicted_increase_in_traffic_by_densifcation[column_inprovement] = (
            df_predicted_increase_in_traffic_by_densifcation[column_inprovement].apply(
                lambda x: 0 if x < 0 else x))
    max_weeks_to_predict = 260
    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
    week_of_the_upgrade = dt.strptime(week_of_the_upgrade + "0", "%Y%U%w")
    index = pd.date_range(week_of_the_upgrade + pd.Timedelta(1, 'W'),
                          periods=max_weeks_to_predict, freq="W")
    df_date = pd.DataFrame({"date": index})
    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
    df_date["week_of_the_upgrade"] = week_of_the_upgrade
    df_date["week_of_the_upgrade"] = df_date.week_of_the_upgrade.dt.strftime("%Y%U")
    df_date["week_of_the_upgrade"] = df_date["week_of_the_upgrade"].apply(int).apply(str)

    df_date["lag_between_the_upgrade"] = list(range(1, 261))

    # Merge dates with file
    df_predicted_increase_in_traffic_by_densifcation_merge = (
        df_predicted_increase_in_traffic_by_densifcation.merge(
            df_date, how="left", on='week_of_the_upgrade'
        ))

    return df_predicted_increase_in_traffic_by_densifcation_merge


@add_logging_info
def create_cluster_from_site_to_model(site_to_model):
    """
    Function to create cluster dataset from site to model

    Parameters
    ----------
    site_to_model: pd.DataFrame

    Returns
    -------
    df_cluster_key: pd.DataFrame

    """
    cols_to_keep = ['site_id', 'neighbour_1', 'neighbour_2']

    # Filter on columns
    df_cluster_key = site_to_model[cols_to_keep]

    # Create cluster key column
    df_cluster_key['cluster_key'] = df_cluster_key.apply(concat_values_for_cluster_key, axis=1)
    return df_cluster_key


def concat_values_for_cluster_key(row):
    """
    The concat_values_for_cluster_key function concatenates values from specific columns of a
    DataFrame row to create a unique cluster key. If the neighbour_2 column is missing (NaN),
    it concatenates site_id and neighbour_1. Otherwise, it concatenates site_id, neighbour_1, and
    neighbour_2.

    Parameters
    ----------
    row : raw
         row from a DataFrame, typically passed using the apply method

    Returns
    -------
    A string representing the concatenated cluster key.
    """
    if pd.isna(row['neighbour_2']):
        return row['site_id'] + '_' + row['neighbour_1']
    return row['site_id'] + '_' + row['neighbour_1'] + '_' + row['neighbour_2']


#@add_logging_info
#def pipeline_modify_structure_for_capacity(site_to_model, prediction):
#    """
#    Function that will rework the format to answer the capacity pipeline achitecture
#
#    Parameters
#    ----------
#    site_to_model: pd.DataFrame
#    prediction: pd.DataFrame
#
#    Return
#    -------
#    df_predicted_increase_in_traffic_by_densifcation_merge: pd.DataFrame
#    """
#    df_predicted_increase_in_traffic_by_densifcation = pd.DataFrame(
#        prediction['site_id'].drop_duplicates(), columns=['site_id'])
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    df_predicted_increase_in_traffic_by_densifcation["week_of_the_upgrade"] = week_of_the_upgrade
#    df_predicted_increase_in_traffic_by_densifcation["bands_upgraded"] = "densification"
#
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        site_to_model[['site_id', 'n_traffic_before']].drop_duplicates(),
#            how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        prediction, how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation[
#        'trafic_improvement'] =(df_predicted_increase_in_traffic_by_densifcation.pred_traffic_data-
#                                 df_predicted_increase_in_traffic_by_densifcation.n_traffic_before)
#    # If trafic improvement is negatif we assign to 0
#    df_predicted_increase_in_traffic_by_densifcation['trafic_improvement'] = \
#    df_predicted_increase_in_traffic_by_densifcation['trafic_improvement'].apply(
#        lambda x: 0 if x < 0 else x)
#
#    max_weeks_to_predict = 260
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    week_of_the_upgrade = dt.strptime(week_of_the_upgrade + "0", "%Y%U%w")
#    index = pd.date_range(week_of_the_upgrade + pd.Timedelta(1, 'W'),
#                          periods=max_weeks_to_predict,freq="W")
#    df_date = pd.DataFrame({"date": index})
#    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = week_of_the_upgrade
#    df_date["week_of_the_upgrade"] = df_date.week_of_the_upgrade.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = df_date["week_of_the_upgrade"].apply(int).apply(str)
#
#    df_date["lag_between_the_upgrade"] = list(range(1, 261))
#
#    # Merge dates with file
#    df_predicted_increase_in_traffic_by_densifcation_merge = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        df_date, how="left", on='week_of_the_upgrade'
#    ))
#
#    return df_predicted_increase_in_traffic_by_densifcation_merge
#
#def pipeline_modify_structure_for_capacity_site(site_to_model, prediction):
#    """
#    Function that will rework the format to answer the capacity pipeline achitecture
#
#    Parameters
#    ----------
#    site_to_model: pd.DataFrame
#    prediction: pd.DataFrame
#
#    Return
#    -------
#    df_predicted_increase_in_traffic_by_densifcation_merge: pd.DataFrame
#    """
#
#    df_predicted_increase_in_traffic_by_densifcation = pd.DataFrame(
#        prediction['site_id'].drop_duplicates(), columns=['site_id'])
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    df_predicted_increase_in_traffic_by_densifcation["week_of_the_upgrade"] = week_of_the_upgrade
#    df_predicted_increase_in_traffic_by_densifcation["bands_upgraded"] = "densification"
#
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        site_to_model[['site_id', 'n_traffic_before']].drop_duplicates(),
#            how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        prediction, how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation[
#        'trafic_improvement'] = df_predicted_increase_in_traffic_by_densifcation.predicted_traffic
#    # If trafic improvement is negatif we assign to 0
#
#    max_weeks_to_predict = 260
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    week_of_the_upgrade = dt.strptime(week_of_the_upgrade + "0", "%Y%U%w")
#    index = pd.date_range(week_of_the_upgrade + pd.Timedelta(1, 'W'),
#                          periods=max_weeks_to_predict,freq="W")
#    df_date = pd.DataFrame({"date": index})
#    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = week_of_the_upgrade
#    df_date["week_of_the_upgrade"] = df_date.week_of_the_upgrade.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = df_date["week_of_the_upgrade"].apply(int).apply(str)
#
#    df_date["lag_between_the_upgrade"] = list(range(1, 261))
#
#    # Merge dates with file
#    df_predicted_increase_in_traffic_by_densifcation_merge = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        df_date, how="left", on='week_of_the_upgrade'
#    ))
#
#    return df_predicted_increase_in_traffic_by_densifcation_merge
#
#
#@add_logging_info
#def pipeline_modify_structure_for_capacity_voice(site_to_model, prediction):
#    """
#    Function that will rework the format to answer the capacity pipeline achitecture
#
#    Parameters
#    ----------
#    site_to_model: pd.DataFrame
#    prediction: pd.DataFrame
#
#    Return
#    -------
#    df_predicted_increase_in_traffic_by_densifcation_merge: pd.DataFrame
#    """
#    df_predicted_increase_in_traffic_by_densifcation = pd.DataFrame(
#        prediction['site_id'].drop_duplicates(), columns=['site_id'])
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    df_predicted_increase_in_traffic_by_densifcation["week_of_the_upgrade"] = week_of_the_upgrade
#    df_predicted_increase_in_traffic_by_densifcation["bands_upgraded"] = "densification"
#
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        site_to_model[['site_id', 'n_voice_before']].drop_duplicates(),
#            how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        prediction, how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation[
#        'trafic_improvement_voice'] = (df_predicted_increase_in_traffic_by_densifcation.pred_voice-
#                                 df_predicted_increase_in_traffic_by_densifcation.n_voice_before)
#
#    max_weeks_to_predict = 260
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    week_of_the_upgrade = dt.strptime(week_of_the_upgrade + "0", "%Y%U%w")
#    index = pd.date_range(week_of_the_upgrade + pd.Timedelta(1, 'W'),
#                          periods=max_weeks_to_predict,freq="W")
#    df_date = pd.DataFrame({"date": index})
#    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = week_of_the_upgrade
#    df_date["week_of_the_upgrade"] = df_date.week_of_the_upgrade.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = df_date["week_of_the_upgrade"].apply(int).apply(str)
#
#    df_date["lag_between_the_upgrade"] = list(range(1, 261))
#
#    # Merge dates with file
#    df_predicted_increase_in_traffic_by_densifcation_merge_voice = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        df_date, how="left", on='week_of_the_upgrade'
#    ))
#
#    return df_predicted_increase_in_traffic_by_densifcation_merge_voice
#
#
#def pipeline_modify_structure_for_capacity_voice_site(site_to_model, prediction):
#    """
#    Function that will rework the format to answer the capacity pipeline achitecture
#
#    Parameters
#    ----------
#    site_to_model: pd.DataFrame
#    prediction: pd.DataFrame
#
#    Return
#    -------
#    df_predicted_increase_in_traffic_by_densifcation_merge: pd.DataFrame
#    """
#    df_predicted_increase_in_traffic_by_densifcation = pd.DataFrame(
#        prediction['site_id'].drop_duplicates(), columns=['site_id'])
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    df_predicted_increase_in_traffic_by_densifcation["week_of_the_upgrade"] = week_of_the_upgrade
#    df_predicted_increase_in_traffic_by_densifcation["bands_upgraded"] = "densification"
#
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        site_to_model[['site_id', 'n_voice_before']].drop_duplicates(),
#            how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        prediction, how="inner", on="site_id"))
#    df_predicted_increase_in_traffic_by_densifcation[
#        'trafic_improvement_voice'] = (
#        df_predicted_increase_in_traffic_by_densifcation.predicted_traffic)
#
#    max_weeks_to_predict = 260
#    week_of_the_upgrade = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
#    week_of_the_upgrade = dt.strptime(week_of_the_upgrade + "0", "%Y%U%w")
#    index = pd.date_range(week_of_the_upgrade + pd.Timedelta(1, 'W'),
#                          periods=max_weeks_to_predict,freq="W")
#    df_date = pd.DataFrame({"date": index})
#    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = week_of_the_upgrade
#    df_date["week_of_the_upgrade"] = df_date.week_of_the_upgrade.dt.strftime("%Y%U")
#    df_date["week_of_the_upgrade"] = df_date["week_of_the_upgrade"].apply(int).apply(str)
#
#    df_date["lag_between_the_upgrade"] = list(range(1, 261))
#
#    # Merge dates with file
#    df_predicted_increase_in_traffic_by_densifcation_merge_voice = (
#        df_predicted_increase_in_traffic_by_densifcation.merge(
#        df_date, how="left", on='week_of_the_upgrade'
#    ))
#
#    return df_predicted_increase_in_traffic_by_densifcation_merge_voice
