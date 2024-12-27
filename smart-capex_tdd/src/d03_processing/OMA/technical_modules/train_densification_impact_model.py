import logging
import os
from datetime import datetime as dt

import haversine as hs
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from scipy import spatial
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.d00_conf.conf import conf
from src.d01_utils.utils import add_logging_info




def compute_distance_between_sites(site):
    """
    Function to compute distance between all sites

    Parameters
    ----------
    site: pd.DataFrame
        Dataset with site's information

    Returns
    -------
    distance: pd.DataFrame
        Dataset with distance between all site

    """
    site_ = site[['longitude', 'latitude', 'site_id']].drop_duplicates()
    all_points = site_[['latitude', 'longitude']].values
    dm1 = spatial.distance.cdist(all_points, all_points, hs.haversine)
    #distance = pd.DataFrame(dm1, index=site['site_id'].values, columns=site['site_id'].values)
    distance = pd.DataFrame(dm1, index=site_['site_id'].values, columns=site_['site_id'].values)
    return distance


def minmax_scale_numerical_value(kpis):
    """
    Scaler numerical data of total_data_traffic_dl_gb and average_prb_load_dl

    Parameters
    ----------
    kpis: pd.DataFrame
        weekly kpis

    Return
    ------
    kpis: pd.dataframe
        kpis scaled
    scaler_feature: scaler
        each scaler
    """
    # Scale the numerical features
    scaler_traf = MinMaxScaler()
    kpis.total_data_traffic_dl_gb = scaler_traf.fit_transform(
        np.array(kpis.total_data_traffic_dl_gb).reshape(-1, 1))
    scaler_prb = MinMaxScaler()
    # kpis.max_prb_load_dl = scaler_prb.fit_transform(np.array(kpis.max_prb_load_dl).reshape(-1, 1))
    kpis.average_prb_load_dl = scaler_prb.fit_transform(
        np.array(kpis.average_prb_load_dl).reshape(-1, 1))
    # kpis.average_prb_load_dl = scaler_prb.transform(np.array(kpis.avg_prb_load_dl).reshape(-1, 1))
    return kpis, scaler_traf, scaler_prb


def label_encode_categorical_features(site):
    """
    Encode categorical features

    Parameters
    ----------
    site: pd.Dataframe
        every site site and their description

    Returns
    -------
    site: pd.Dataframe
        site encoded
    le_features: encoder
        each label encoder for each feature
    """
    # Label encode the categorical features
    le_region = preprocessing.LabelEncoder()
    le_ville = preprocessing.LabelEncoder()
    le_commune = preprocessing.LabelEncoder()
    le_province = preprocessing.LabelEncoder()
    site.region = le_region.fit_transform(site.region)
    site.ville = le_ville.fit_transform(site.ville)
    site.commune = le_commune.fit_transform(site.commune)
    site.province = le_province.fit_transform(site.province)
    return site, le_region, le_ville, le_commune, le_province


def compute_max_distance_neighbour(df_distance, distance_neighbour, max_neighbour):
    """
    Function to compute max distance between all neighbour
    Parameters
    ----------
    df_distance: pd.DataFrame
    distance_neighbour: int
    max_neighbour: int
    Returns
    -------
    cluster_neighbours: pd.DataFrame
    """
    distance_vector = df_distance[(df_distance > 0) & (df_distance < distance_neighbour)]
    site_id_to_index = {}
    for site_id in distance_vector.columns:
        dist_order = distance_vector.loc[site_id]
        try:
            if dist_order.sort_values(ascending=True).sum() != 0:
                dist_order = dist_order.sort_values(ascending=True)
                site_id_to_index[site_id] = list(dist_order[:max_neighbour].index)
            else:
                pass
        except TypeError:
            print(site_id)
    cluster_neighbours = pd.DataFrame(site_id_to_index)
    cluster_neighbours = pd.concat([cluster_neighbours.loc[0], cluster_neighbours.loc[1]],
                                   axis=1)
    cluster_neighbours.columns = ['neighbour_' + str(x + 1) for x in range(max_neighbour)]
    return cluster_neighbours


def get_neighbours(site_data):
    """
    Get the nearest desired number of neighbours

    Parameters
    ----------
    site_data: pd.Dataframe
        data containing every site
    Return
    ------
    cln_: pd.dataframe
        each new site and its nearest neighbours
    """
    # Compute distance between site
    distance = compute_distance_between_sites(site_data)
    cln = compute_max_distance_neighbour(df_distance=distance, distance_neighbour=10,
                                         max_neighbour=2)
    new_site = site_data[~site_data['date_site'].isna()]
    #new_site = new_site.drop('Site', axis=1)
    cln = cln.reset_index()
    cln.columns = ['site_id', 'neighbour_1', 'neighbour_2']
    cln_ = cln.merge(new_site, on='site_id', how='inner')
    cln_ = cln_[['site_id', 'neighbour_1', 'neighbour_2', 'date_site',
                 'commune', 'ville', 'province', 'region']].drop_duplicates()
    return cln_


def compute_kpis_before_after_deployment(data):
    """
    Compute the kpi before and after

    Parameters
    ----------
    data: pd.Dataframe
        aggregated data on every site and its neighbours

    Return
    -------
    kpi_before: pd.Dataframe
        kpi before
    kpi_after: pd.Dataframe
        kpi after

    """
    # Compare the date with date_site to attribuate kpi before, after or nothing
    data['kpi_features'] = data.apply(lambda x:
                                      np.where(
                                          ((x['date_site'] + relativedelta(weeks=-9) < x.date) and (
                                                  x.date < x['date_site'])),
                                          'kpi_before',
                                          np.where(
                                              ((x.date > x['date_site'] + relativedelta(weeks=3))
                                               and (x.date < x['date_site'] + relativedelta(
                                                          weeks=12))),
                                              'kpi_after', 'nothing')
                                      ), axis=1)
    kpi_before = data[data['kpi_features'] == 'kpi_before']

    # Compute mean of each kpi for each site id
    kpi_before = kpi_before.groupby('site_id').agg(
        {'n_traffic': np.nanmean, 'n_voice': np.nanmean, 'n_avgprb': np.nanmean,
         'n_avgthrput': np.nanmean,
         'n_avgusr': np.nanmean}).reset_index()


    kpi_before.columns = ['site_id', 'n_traffic_before', 'n_voice_before', 'n_avgprb_before',
                          'n_avgthrput_before','n_avgusr_before']

    # Keep kpi_after and group by site_id
    kpi_after = data[data['kpi_features'] == 'kpi_after']
    kpi_after = kpi_after.groupby('site_id').agg({'n_traffic': np.nanmean, 'n_avgprb': np.nanmean,
                                                  'n_voice': np.nanmean}).reset_index()

    # Clean with fill na, and keep kpi > 0
    kpi_after.columns = ['site_id', 'n_traffic_after', 'n_avgprb_after', 'n_voice_after']
    kpi_after.fillna(0, inplace=True)
    kpi_after = kpi_after[kpi_after.n_avgprb_after > 0]
    kpi_after = kpi_after[kpi_after.n_traffic_after > 0]
    # kpi_after = kpi_after[kpi_after.n_maxprb_after > 0]
    return kpi_before, kpi_after


def prepare_train_improvement_model_deployment(cln_, kpis):
    """
    Prepare and merged distance of every site and their corresponding kpis

    Parameters
    ----------
    cln_: pd.Dataframe
        distance between site
    kpis: pd.Dataframe
        weekly kpis

    Return
    ------
    new_sites: pd.Dataframe
        each site and neighbours and their corresponding kpis (unaggregated)
    """
    # Add historical traffic by site
    kpis_site = kpis.groupby(['site_id', 'date']).agg({'total_data_traffic_dl_gb': 'sum',
                                                       'total_voice_traffic_kerlangs': 'sum',
                                                       'average_prb_load_dl': 'sum',
                                                       # 'max_prb_load_dl': 'sum',
                                                       'average_throughput_dl_kbps': 'sum',
                                                       # 'average_throughput_user_dl_kbps': 'sum',
                                                       # 'max_throughput_user_dl_kbps': 'sum',
                                                       # 'max_active_users': 'sum',
                                                       'average_active_users': 'sum',
                                                       }).reset_index()
    kpis_site['date'] = kpis_site['date'].astype(str)
    kpis_site['date'] = kpis_site['date'].str.split('/').str[0]
    kpis_site['date'] = pd.to_datetime(kpis_site.date, format="%Y-%m-%d")

    # Traffic for neighbour_1
    new_sites = cln_[['site_id', 'neighbour_1', 'neighbour_2', 'date_site','commune', 'ville',
                      'province', 'region']] \
        .merge(kpis_site, left_on='neighbour_1', right_on='site_id', how='left',
               validate='many_to_many')

    # Traffic for its neighbour 2
    new_sites = new_sites.merge(kpis_site, left_on='neighbour_2', right_on='site_id', how='right',
                                suffixes=['_n1', '_n2'])
    # Cleaning new sites (drop some columns)
    new_sites = cleaning_new_sites(new_sites)

    # Rename columns
    new_sites.columns = ['site_id', 'neighbour_1', 'neighbour_2', 'date_site',
                         'commune', 'ville', 'province', 'region', 'date',
                         'n1_traffic', 'n1_avgvoice', 'n1_avgprb', 'n1_avgthrput', 'n1_avgusr',
                         'n2_traffic','n2_avgvoice', 'n2_avgprb', 'n2_avgthrput', 'n2_avgusr', ]

    # Calcul mean of n1, n2 for each kpi
    new_sites['n_traffic'] = new_sites.apply(lambda x: np.sum([x.n1_traffic, x.n2_traffic]),
                                             axis=1)
    new_sites['n_voice'] = new_sites.apply(lambda x: np.sum([x.n1_avgvoice, x.n2_avgvoice]),
                                             axis=1)
    new_sites['n_avgprb'] = new_sites.apply(lambda x: np.sum([x.n1_avgprb, x.n2_avgprb]),
                                            axis=1)
    # new_sites['n_maxprb'] = new_sites.apply(lambda x: np.nanmean([x.n1_maxprb, x.n2_maxprb]),
    # axis=1)
    new_sites['n_avgthrput'] = new_sites.apply(lambda x: np.sum(
        [x.n1_avgthrput, x.n2_avgthrput]), axis=1)
    # new_sites['n_maxthrput'] = new_sites.apply(
    # lambda x: np.nanmean([x.n1_maxthrput, x.n2_maxthrput]), axis=1)
    new_sites['n_avgusr'] = new_sites.apply(lambda x: np.sum([x.n1_avgusr, x.n2_avgusr]),
                                            axis=1)
    # new_sites['n_maxusr'] = new_sites.apply(lambda x: np.nanmean([x.n1_maxusr, x.n2_maxusr]),
    # axis=1)

    # Filter on new column and clean columns
    new_sites = new_sites[['site_id', 'neighbour_1', 'neighbour_2', 'date_site',
                           'commune', 'ville', 'province', 'region', 'date',
                           'n_traffic', 'n_voice', 'n_avgprb', 'n_avgthrput', 'n_avgusr']]
    new_sites = new_sites.sort_values(['site_id', 'date'], ascending=True)
    new_sites = new_sites.drop_duplicates()
    new_sites['date_site'] = new_sites['date_site'].apply(convert_to_date)
    return new_sites


def convert_to_date(x):
    """
    The convert_to_date function attempts to convert a string to a datetime object using the format
    "%Y-%m-%d". If the conversion fails or the input is not a string,
    it returns None or the original input, respectively

    Parameters
    ----------
    x : str
        The input value which can be a string representing a date or any other type

    Returns
    -------
    x : datetime or None
        A datetime object if the input string is successfully converted.
        None if the input string cannot be converted.
        The original input if it is not a string
    """
    if isinstance(x, str):
        try:
            return dt.strptime(x, "%Y-%m-%d")
        except ValueError:
            return None
    return x


def cleaning_new_sites(new_sites):
    """
    The cleaning_new_sites function filters and cleans the new_sites DataFrame by ensuring that the
    dates for neighbour_1 and neighbour_2 match, and then drops unnecessary columns to streamline
    the data.

    Parameters
    ----------
    new_sites: pd.DataFrame
        A DataFrame containing site and neighbor information along with their KPIs.
    Returns
    -------
    new_sites: pd.DataFrame
        A cleaned DataFrame with only the necessary columns and rows where the dates for
        neighbour_1 and neighbour_2 match.
    """
    new_sites = new_sites[new_sites.date_n1 == new_sites.date_n2]
    new_sites.drop('site_id', axis=1, inplace=True)
    new_sites.drop('site_id_y', axis=1, inplace=True)
    new_sites.drop('date_n2', axis=1, inplace=True)
    return new_sites


def create_site_to_model(new_sites, kpi_before_after):
    """

    Parameters
    ----------
    new_sites: pd.Dataframe
    kpi_before_after: pd.Dataframe

    Returns
    -------
    site_to_model: pd.Dataframe
    """
    # Link traffic with site_id
    columns_to_select = ['site_id', 'region', 'ville', 'commune', 'province']
    sites_to_model = new_sites[columns_to_select]
    sites_to_model = kpi_before_after.merge(sites_to_model, on='site_id', how='left')
    sites_to_model = sites_to_model.drop_duplicates()
    return sites_to_model


def xgboost_new_site(site_to_model_, y_column, x_columns):
    """
    Train the model - xgboost - with a 10% test data and output the MAPE and R2 of the model.
    The final model is trained on the full dataset with the best parameters

    Parameters
    ----------
    site_to_model_: pd.Dataframe
        site to model
    y_column: str
        The target column to model
    x_columns: list
        List containing every features to train with
    scaler: sklearn scaler
        the corresponding scaler for the target column (used to output model metrics)

    Return
    ------
    rf_model_final: pd.Dataframe
        Final model trained on every data
    MAPE: pd.Dataframe
        MAPE on the test/train model
    r2: pd.Dataframe
        r2 on the test/train model
    """
    # Nan to 0
    site_to_model_ = site_to_model_.fillna(0)
    # Regression model for traffic after
    # 1. Splitting between train and test
    x = np.array(site_to_model_[x_columns])
    y = np.array(site_to_model_[y_column])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=123)
    # 2. Training the model
    xgb_model = xgb.XGBRegressor(random_state=214)
    # Define parameters to test
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [5, 10, 15, 20]
    }
    if len(x_train) == 1:
        x_train = np.vstack([x_train, x_train])
        y_train = np.vstack([y_train, y_train])
    # Gridsearch over the parameters
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=conf['TRAIN']['CROSS_VALIDATION'], scoring='r2')
    grid_search.fit(x_train, y_train)
    # Find best parameters
    best_params = grid_search.best_params_
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    # Prediction
    predictions_test = grid_search.predict(x_test)
    # MAE
    mae = mean_absolute_error(y_test, predictions_test)
    print(f"MAE of the base model: {mae:.3f}")
    # MAPE
    mape = mean_absolute_percentage_error(y_test, predictions_test)
    print(f"MAPE of the base model: {mape:.3f}")
    # R2
    r2 = r2_score(y_test, predictions_test)
    print(f"R2 of the base model: {r2:.3f}")

    # Model on all data with good param
    xgb_model_final = xgb.XGBRegressor(random_state=214, n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           min_samples_split=best_params['min_samples_split'])
    xgb_model_final.fit(x, y)

    imp = {"VarName": site_to_model_[x_columns].columns,
           'Importance': xgb_model_final.feature_importances_}
    print('Feature Importance')
    print(pd.DataFrame(imp))
    # Rebuild final table
    predictions_train = grid_search.predict(x_train)


    df_x = pd.concat([pd.DataFrame(x_train, columns = x_columns),
                      pd.DataFrame(x_test, columns = x_columns)], axis=0)
    df_y_train = pd.DataFrame(y_train, columns = [y_column])
    df_y_train['source'] = 'train'
    df_y_test = pd.DataFrame(y_test, columns = [y_column])
    df_y_test['source'] = 'test'
    df_y = pd.concat([df_y_train, df_y_test], axis = 0)
    df_prediction = pd.concat([pd.DataFrame(predictions_train, columns=["prediction"]),
                               pd.DataFrame(predictions_test, columns=["prediction"])], axis=0)
    df_y_withpred = pd.concat([df_y, df_prediction], axis = 1)
    df_analyse = pd.concat([df_x, df_y_withpred], axis = 1)
    df_analyse['mape'] = mape
    df_analyse['mae'] = mae
    df_analyse['r2'] = r2
    return xgb_model_final, mape, r2, df_analyse


def random_forest_new_site(site_to_model_, y_column, x_columns, scaler):
    """
    Train the model - RandomForest - with a 10% test data and output the mape and R2 of the model.
    The final model is trained on the full dataset with the best parameters

    Parameters
    ----------
    site_to_model_: pd.Dataframe
        site to model
    y_column: str
        The target column to model
    x_columns: list
        List containing every features to train with
    scaler: sklearn scaler
        the corresponding scaler for the target column (used to output model metrics)

    Return
    ------
    rf_model_final: pd.Dataframe
        Final model trained on every data
    mape: pd.Dataframe
        mape on the test/train model
    r2: pd.Dataframe
        r2 on the test/train model
    """
    # Nan to 0
    site_to_model_ = site_to_model_.fillna(0)
    # Regression model for traffic after
    # 1. Splitting between train and test
    x_df = site_to_model_[x_columns]
    x = np.array(x_df)
    y = np.array(site_to_model_[y_column])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10,
                                                        random_state=123)
    # 2. Training the model
    rf_model = RandomForestRegressor(random_state=214)
    # Define parameters to test
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [5, 10, 15, 20]
    }
    # Gridsearch over the parameters
    best_params, grid_search = compute_grid_search(param_grid, rf_model, x_train, y_train)
    # Prediction
    predictions_test = grid_search.predict(x_test)
    #predictions_test = scaler.inverse_transform(predictions_test.reshape(-1, 1))
    #y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # mae
    mae, mape, r2 = print_metric(predictions_test, y_test)

    # Model on all data with good param
    rf_model_final = RandomForestRegressor(random_state=214,
                                           n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           min_samples_split=best_params['min_samples_split'])
    rf_model_final.fit(x, y)

    imp = {"VarName": x_df.columns, 'Importance': rf_model_final.feature_importances_}
    print('Feature Importance')
    print(pd.DataFrame(imp))
    # Rebuild final table
    predictions_train = grid_search.predict(x_train)
    print(scaler)
    #predictions_train = scaler.inverse_transform(predictions_train.reshape(-1, 1))
    #y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    df_analyse = build_final_table(mae, mape, predictions_test, predictions_train, r2, x_columns,
                                   x_test, x_train, y_column, y_test, y_train)
    return rf_model_final, mape, r2, df_analyse


def print_metric(predictions_test, y_test):
    mae = mean_absolute_error(y_test, predictions_test)
    print(f"mae of the base model: {mae:.3f}")
    # mape
    mape = mean_absolute_percentage_error(y_test, predictions_test)
    print(f"mape of the base model: {mape:.3f}")
    # R2
    r2 = r2_score(y_test, predictions_test)
    print(f"R2 of the base model: {r2:.3f}")
    return mae, mape, r2


def build_final_table(mae, mape, predictions_test, predictions_train, r2, x_columns, x_test,
                      x_train, y_column, y_test, y_train):
    """
    The build_final_table function consolidates training and testing data, predictions,
    and evaluation metrics into a single DataFrame for further analysis.

    Parameters
    ----------
    mae: float
        Mean Absolute Error of the model.
    mape :float
        Mean Absolute Percentage Error of the model.
    predictions_test: series
        Predictions made on the test dataset
    predictions_train: series
        Predictions made on the train dataset
    r2: float
        R-squared value of the model.
    x_columns:list
        List of feature column names
    x_test: pd.DataFrame
        Test dataset features
    x_train: pd.DataFrame
        Train dataset features
    y_column: str
        Name of the target column.
    y_test: series
        Actual target values for the test dataset.
    y_train: series
        Actual target values for the train dataset.

    Returns
    -------
    df_analyse: pd.DataFrame
        A DataFrame that includes features, actual target values, predictions,
        and evaluation metrics for both training and testing datasets

    """
    df_x = pd.concat(
        [pd.DataFrame(x_train, columns=x_columns), pd.DataFrame(x_test, columns=x_columns)],
        axis=0)
    df_y_train = pd.DataFrame(y_train, columns=[y_column])
    df_y_train['source'] = 'train'
    df_y_test = pd.DataFrame(y_test, columns=[y_column])
    df_y_test['source'] = 'test'
    df_y = pd.concat([df_y_train, df_y_test], axis=0)
    df_prediction = pd.concat([pd.DataFrame(predictions_train, columns=["prediction"]),
                               pd.DataFrame(predictions_test, columns=["prediction"])], axis=0)
    df_y_withpred = pd.concat([df_y, df_prediction], axis=1)
    df_analyse = pd.concat([df_x, df_y_withpred], axis=1)
    df_analyse['mape'] = mape
    df_analyse['mae'] = mae
    df_analyse['r2'] = r2
    return df_analyse


def compute_grid_search(param_grid, rf_model, x_train, y_train):
    """


    Parameters
    ----------
    param_grid
    rf_model
    x_train
    y_train

    Returns
    -------

    """
    if len(x_train) == 1:
        x_train = np.vstack([x_train, x_train])
        y_train = np.vstack([y_train, y_train])
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=conf['TRAIN']['CROSS_VALIDATION'], scoring='r2')
    grid_search.fit(x_train, y_train)
    # Find best parameters
    best_params = grid_search.best_params_
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    return best_params, grid_search


@add_logging_info
def pipeline_train_model_newsite_deployment(kpis, site):
    """
    Train the impact model of a new site on its neighbours. Site needs to be improved with the file
     containing thedeployment history. It will compute the distance between site,
     find the 2 nearest neighbour, compute weekly kpisand find the average traffic before and
     average traffic after a new site appears. Then it will create 2 models,one for traffic
     the other for prb, and save every file, scaler, encoder and model in the d05 directory

    Parameters
    ----------
    kpis: pd.Dataframe
        Historical weekly data
    site: pd.Dataframe
        Every sites

    Return
    -------
    site_to_model: pd.Dataframe
        The site used for modeling
    new_sites: pd.Dataframe
        the new sites and their kpis unaggregated
    kpi_before_after: pd.Dataframe
        The aggregated kpis before and after upgrade
    """
    # Scale the numerical values
    #kpis, scaler_traf, scaler_prb = minmax_scale_numerical_value(kpis)

    # Encode Categorical Value
    site, le_region, le_ville, le_commune, le_province = label_encode_categorical_features(site)

    # Extract number of 4G cells for each site
    nb_cell_4g = extract_nb_cell_4g(site)

    # Get the nearest desired number of neighbours
    cln_ = get_neighbours(site)

   ## Save and read neighbours dataset
    cln_.to_csv(os.path.join(conf['PATH']['PROCESSED_DATA'], 'df_neighbour_' + conf['USE_CASE'] +
                            ".csv"), sep='|', index=False)
    cln_ = pd.read_csv(os.path.join(conf['PATH']['PROCESSED_DATA'], 'df_neighbour_') +
                       conf['USE_CASE'] + ".csv", sep='|')

    # Prepare and merged distance of every site and their corresponding kpis
    new_sites = prepare_train_improvement_model_deployment(cln_, kpis)

    # Compute kpi before and after
    kpi_before, kpi_after = compute_kpis_before_after_deployment(new_sites)

    # Merge kpi before with kpi after
    kpi_before_after = kpi_before.merge(kpi_after, on='site_id', how="inner")

    # Link traffic with site id
    site_to_model = create_site_to_model(new_sites, kpi_before_after)
    site_to_model = site_to_model.merge(nb_cell_4g, on='site_id', how='left')
    col_to_model = ['n_traffic_before', 'n_avgprb_before',
                    'n_avgthrput_before', 'n_avgusr_before']

    # Model 1 congestion: prb_load on neighbouring site
    #scaler_prb = None
    #scaler_traf = None

    # Trail all models (RF, XGBoost) on different kpis ...
    (df_analyse_prb, df_analyse_traf, mape_prb, mape_traf, r2_prb, r2_traf, rf_neighbour_prb,
     rf_neighbour_traf, xgb_neighbour_traf, mape_traf_xgb, r2_traf_xgb,
            df_analyse_traf_xgb) = train_all_models(
        col_to_model, None, None, site_to_model)
    dict_result = {
        'mape_prb': mape_prb,
        'r2_prb': r2_prb,
        'mape_traf': mape_traf,
        'r2_traf': r2_traf,
        'xgb_neighbour_traf': xgb_neighbour_traf,
        'mape_traf_xgb': mape_traf_xgb,
        'r2_traf_xgb': r2_traf_xgb
    }
    col_to_model = ['n_avgprb_before',
                    'n_avgthrput_before', 'n_avgusr_before', 'n_voice_before']
    target_column = 'n_voice_after'  # 'n_traffic_after', 'n_avgprb_after', 'n_maxprb_after'
    rf_neighbour_voice, _, _, df_analyse_voice = (
        random_forest_new_site(site_to_model,
                               target_column,
                               col_to_model,
                               scaler=None))

    # Save all models and dataframe
    logging.info(dict_result)
    save_models_and_encoders_and_df(cln_, df_analyse_prb, df_analyse_traf, df_analyse_traf_xgb,
                                    df_analyse_voice, kpi_before_after, le_commune, le_province,
                                    le_region, le_ville, new_sites, rf_neighbour_prb,
                                    rf_neighbour_traf, rf_neighbour_voice,
                                    site_to_model, xgb_neighbour_traf)

    return site_to_model, new_sites, kpi_before_after


def save_models_and_encoders_and_df(cln_, df_analyse_prb, df_analyse_traf, df_analyse_traf_xgb,
                                    df_analyse_voice, kpi_before_after, le_commune, le_province,
                                    le_region, le_ville, new_sites, rf_neighbour_prb,
                                    rf_neighbour_traf, rf_neighbour_voice,
                                    site_to_model, xgb_neighbour_traf):
    joblib.dump(rf_neighbour_prb,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_prb_neighbour.joblib'))
    joblib.dump(rf_neighbour_traf,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_traffic_neighbour.joblib'))
    joblib.dump(rf_neighbour_voice,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_voice_neighbour.joblib'))
    # Test d'enregistrer le modèle xgv
    joblib.dump(xgb_neighbour_traf,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'rf_traffic_neighbour.joblib'))
    #joblib.dump(scaler_traf,
    #            os.path.join(conf['PATH']['MODELS_OUTPUT'], 'scaler_traffic.joblib'),
    #            compress=9)
    #joblib.dump(scaler_prb,
    #            os.path.join(conf['PATH']['MODELS_OUTPUT'], 'scaler_prb.joblib'),
    #            compress=9)
    joblib.dump(le_region,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_region.joblib'),
                compress=9)
    joblib.dump(le_ville,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_ville.joblib'),
                compress=9)
    joblib.dump(le_commune,
                os.path.join(conf['PATH']['MODELS_OUTPUT'], 'encoder_commune.joblib'),
                compress=9)
    joblib.dump(le_province, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                          'encoder_province.joblib'), compress=9)
    site_to_model.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'site_to_model.csv'), sep='|',
        index=False)
    new_sites.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'new_sites.csv'), sep='|',
                     index=False)
    kpi_before_after.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'kpi_before_after.csv'),
                            sep='|', index=False)
    cln_.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'distance_cluster.csv'), sep='|',
                index=False)
    df_analyse_prb.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'df_analyse_prb_' + conf['USE_CASE'] + '.csv'),
        sep='|', index=False)
    df_analyse_traf.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'df_analyse_traf_' + conf['USE_CASE'] + '.csv'),
        sep='|', index=False)
    df_analyse_traf_xgb.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'df_analyse_traf_xgb_' + conf['USE_CASE'] +
                     '.csv'), sep='|', index=False)
    joblib.dump(rf_neighbour_prb, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                               'rf_prb_neighbour.joblib'))
    joblib.dump(rf_neighbour_traf,
                os.path.join(conf['PATH']['MODELS_OUTPUT'],
                             'rf_traffic_neighbour.joblib'))
    #joblib.dump(scaler_traf, os.path.join(conf['PATH']['MODELS_OUTPUT'],
    #                                      'scaler_traffic.joblib'),
    #            compress=9)
    #joblib.dump(scaler_prb, os.path.join(conf['PATH']['MODELS_OUTPUT'], 'scaler_prb.joblib'),
    #            compress=9)
    joblib.dump(le_region, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                        'encoder_region.joblib'),
                compress=9)
    joblib.dump(le_ville, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                       'encoder_ville.joblib'),
                compress=9)
    joblib.dump(le_commune, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                         'encoder_commune.joblib'),
                compress=9)
    joblib.dump(le_province, os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                          'encoder_province.joblib'),
                compress=9)
    site_to_model.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'site_to_model.csv'), sep='|',
        index=False)
    new_sites.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'new_sites.csv'),
                     sep='|', index=False)
    kpi_before_after.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'kpi_before_after.csv'), sep='|',
        index=False)
    cln_.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'], 'distance_cluster.csv'),
                sep='|', index=False)
    df_analyse_prb.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'],
                     'df_analyse_prb_' + conf['USE_CASE'] + '.csv'),
        sep='|', index=False)
    df_analyse_traf.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'],
                     'df_analyse_traf_' + conf['USE_CASE'] + '.csv'),
        sep='|', index=False)
    df_analyse_voice.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'df_analyse_voice_' + conf['USE_CASE'] +
                     '.csv'), sep='|', index=False)


def extract_nb_cell_4g(site):
    """
    For Each site we extract the number of 4G cells
    Parameters
    ----------
    site: pd.DataFrame
        Dataset with site's information

    Returns
    -------
    nb_cell_4g: pd.DataFrame
        Dataset with site id and number of 4G cells
    """
    # Filter on 4G Cell tech
    site_4g = site[site.cell_tech == '4G']

    # For each site count nb of 4g cell
    nb_cell_4g = site_4g.groupby('site_id').agg({'cell_name': 'count'}).reset_index()
    nb_cell_4g.columns = ['site_id', 'nb_cell_4G']
    return nb_cell_4g


def train_all_models(col_to_model, scaler_prb, scaler_traf, site_to_model):
    scaler = scaler_prb  # scaler_traf, scaler_prb, scaler_thrput, scaler_acusr
    target_column = 'n_avgprb_after'  # 'n_traffic_after', 'n_avgprb_after', 'n_maxprb_after'
    rf_neighbour_prb, mape_prb, r2_prb, df_analyse_prb = random_forest_new_site(site_to_model,
                                                                                target_column,
                                                                                col_to_model,
                                                                                scaler=scaler)
    # Model 2 traffic: traffic_dl on neighbouring site
    scaler = scaler_traf
    target_column = 'n_traffic_after'  # 'n_traffic_after', 'n_avgprb_after', 'n_maxprb_after'
    rf_neighbour_traf, mape_traf, r2_traf, df_analyse_traf = random_forest_new_site(site_to_model,
                                                                                    target_column,
                                                                                    col_to_model,
                                                                                    scaler=scaler)
    scaler = None
    target_column = 'n_traffic_after'
    xgb_neighbour_traf, mape_traf_xgb, r2_traf_xgb, df_analyse_traf_xgb = xgboost_new_site(
        site_to_model,
        target_column,
        col_to_model)

    return (df_analyse_prb, df_analyse_traf, mape_prb, mape_traf, r2_prb, r2_traf,
            rf_neighbour_prb, rf_neighbour_traf, xgb_neighbour_traf, mape_traf_xgb, r2_traf_xgb,
            df_analyse_traf_xgb)


def compute_date_deployment(kpis, site, site_densif):
    """
    The compute_date_deployment function processes KPI data and site densification
    data to determine the deployment date of each site based on traffic data.
    It returns an updated site DataFrame with the deployment dates merged.

    Parameters
    ----------
    kpis: pd.DataFrame
        DataFrame containing KPI data with columns date, site_id, and total_data_traffic_dl_gb
    site: pd.DataFrame
        DataFrame containing site information with at least a site_id column
    site_densif: pd.DataFrame
        DataFrame containing site densification data with a Site column

    Returns
    -------
    site: pd.DataFrame
       An updated site DataFrame with an additional date_site column indicating
       the deployment date for each site
    """
    kpis["date"] = pd.to_datetime(kpis.date, format="%Y-%m-%d")
    kpis["year"] = kpis["date"].apply(lambda x: x.year)
    kpis.date = kpis.date.dt.to_period(freq="W")
    site_densif = site_densif.rename(columns={'Site': 'site_id'})
    deployment_history = {"site_id": [], "deployment_date": []}
    for s in site_densif.site_id.to_list():

        traffic_data = kpis[(kpis.site_id == s)]  # & (traffic.year == y)]

        if len(traffic_data) != 0:
            min_date = traffic_data.date.min()
            max_date = traffic_data.date.max()

            agg_date = traffic_data[["total_data_traffic_dl_gb", "date"]].groupby("date").sum()
            idx = pd.period_range(min_date, max_date, freq="W")
            traffic_data_reindex = agg_date.reindex(idx, fill_value=0)

            if len(traffic_data_reindex[
                       traffic_data_reindex['total_data_traffic_dl_gb'] == 0]) != 0:
                deployment_date = traffic_data_reindex[
                                      traffic_data_reindex['total_data_traffic_dl_gb'] == 0].tail(
                    1).index.item() + pd.offsets.Week(1, weekday=6)
            else:
                deployment_date = traffic_data_reindex.head(1).index.item()

            deployment_history["site_id"].append(s)
            deployment_history["deployment_date"].append(deployment_date)
    deployment = pd.DataFrame.from_dict(deployment_history)
    deployment["deployment_date"] = deployment["deployment_date"].astype(str)
    deployment["date_site"] = deployment["deployment_date"].str.split('/').str[0]
    # Merge site file with deployment file
    site = site.merge(deployment[['site_id', 'date_site']], on='site_id',
                      how='left')
    return site
