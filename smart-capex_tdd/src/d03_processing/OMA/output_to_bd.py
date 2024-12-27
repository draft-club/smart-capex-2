"""Module output_to_bd.py"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import write_sql_database



def prepare_forecast_to_db(df_predicted_traffic_kpis, site):
    """
    The prepare_forecast_to_db function processes forecasted traffic KPI data by merging it with
    site information, transforming date formats, and adding new columns to prepare the data for
    database insertion.

    Parameters
    ----------
    df_predicted_traffic_kpis: pd.DataFrame
        A pandas DataFrame containing forecasted traffic KPIs.
    site: pd.DataFrame
        A pandas DataFrame containing site information.

    Returns
    -------
    df_predicted_traffic_kpis: pd.DataFrame
        A pandas DataFrame with processed forecasted traffic KPIs, ready for database insertion.

    """
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.drop('cell_band', axis=1)
    df_predicted_traffic_kpis.date = df_predicted_traffic_kpis.date.astype('datetime64[ns]')
    df_sites_region = site[['site_id', 'region', 'cell_band']].drop_duplicates()
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.merge(df_sites_region,
                                                                how='left', on='site_id')
    df_predicted_traffic_kpis['month_simple'] = (
        df_predicted_traffic_kpis.date.apply(lambda x: f"{x.month:02d}"))
    df_predicted_traffic_kpis['month'] = df_predicted_traffic_kpis.year.astype(str) + \
                                         df_predicted_traffic_kpis.month_simple.astype(str)
    df_predicted_traffic_kpis["all"] = "all"
    return df_predicted_traffic_kpis


def getjson_feature(df: pd.DataFrame, period="week_period", key="site_id"):
    """
    The getjson_feature function processes a DataFrame to aggregate data based on specified periods
    and keys, and then renames and adds columns to standardize the output format.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data to be processed.
    period: str
        A string representing the period column to group by (default is "week_period").
    key: str
        A string representing the key column to group by (default is "site_id").

    Returns
    -------
    df: pd.DataFrame
        A pandas DataFrame with aggregated data, renamed columns, and an additional "type_period"
        column.

    """
    df = df.groupby([period, key]).agg({'total_data_traffic_dl_gb': sum,
                                        'total_data_traffic_ul_gb': sum}).reset_index()
                                        #'average_throughput_user_dl_kbps': np.mean,
                                        #'average_throughput_user_ul_kbps': np.mean,
                                        #'cell_occupation_ul_percentage': np.mean,
                                       # 'cell_occupation_dl_percentage': np.mean,
                                        #'average_number_of_users_in_queue': np.mean,
                                        #'average_number_of_users_ul': np.mean}).reset_index()
    df.rename(columns={period: "period", key: "id"}, inplace=True)
    df["type_period"] = period
    return df



def getband_or_tech(df, config):
    """
    The getband_or_tech function processes a DataFrame by grouping it based on specified columns,
    calculating the mean of a particular value, and then restructuring the data into a new DataFrame
    with JSON-formatted records for each unique key and period combination.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data to be processed
    config : dict
        A dictionary containing configuration parameters such as groupby_cols, value, key,
        and period.

    Returns
    -------
    A pandas DataFrame with columns "id", "period", and the specified value,
    containing JSON-formatted records for each unique key and period combination.

    """
    print(df, config)
    df = df.groupby(config['groupby_cols'])[[config['value']]].mean().reset_index()
    ids = df[config['key']].unique()
    df.rename(columns={config['value']: "value"}, inplace=True)
    results = []
    for id_ in ids:
        inter = df[df[config['key']] == id_].reset_index()
        periods = inter[config['period']].unique()
        for p in periods:
            value = json.loads(
                inter[inter[config['period']] == p][[config['groupby_cols'][0],
                                                     "value"]].to_json(orient="records"))
            results.append([id_, p, value])
    return pd.DataFrame(results, columns=["id", "period", config['value']])


def pipeline_format_data_to_db(df_forecast):
    """
    The pipeline_format_data_to_db function processes a forecast DataFrame by aggregating data based
     on specified keys and periods, and then merges additional features related to technology and
     band. The final DataFrame is prepared for database insertion by standardizing the format and
     concatenating results.

    Parameters
    ----------
    df_forecast: pd.DataFrame
        A pandas DataFrame containing forecast data to be processed.

    Returns
    -------
    df_pandas: pd.DataFrame
        A pandas DataFrame with aggregated and merged data, ready for database insertion.
    """
    # list of features for aggregation
    keys = ["site_id", "region", "all"]
    types = ["site", "region", "all"]
    # list of periods
    periods = ["week_period", "month", "year"]
    df_pandas = pd.DataFrame()

    for i, key in enumerate(keys):
        for period in periods:
            feature_results = getjson_feature(df_forecast, key=key, period=period)
            feature_results.period = feature_results.period.astype(str)
            by = "cell_tech"
            config = {'groupby_cols': [key, period, by],
                      'value': "average_throughput_user_dl_kbps",
                      'key': key,
                      'period': period}
            techs = getband_or_tech(df_forecast,config)
            #techs = getband_or_tech(df_forecast, period=period, key=key, by="cell_tech",
            #                        value="average_throughput_user_dl_kbps", col="debit_tech")
            techs.period = techs.period.astype(str)
            band = getband_or_tech(df_forecast, config)
            #band = getband_or_tech(df_forecast, period=period, key=key, by="cell_band",
            #                       value="average_throughput_user_dl_kbps", col="debit_band")
            band.period = band.period.astype(str)
            feature_results = feature_results.merge(techs, on=["id", "period"], how="left")
            feature_results = feature_results.merge(band, on=["id", "period"], how="left")

            # to_pandas
            feature_results['type_site'] = types[i]
            df_pandas = pd.concat([df_pandas, feature_results], axis=0)
    df_pandas.drop('debit_band', axis=1, inplace=True)
    df_pandas.drop('debit_tech', axis=1, inplace=True)
    return df_pandas


def change_randim_congestion(congestion):
    """
    Changes the randim congestion

    Parameters
    ----------
    congestion: pd.DataFrame

    Returns
    -------
    congestion_: pd.DataFrame
    """
    congestion_ = congestion.drop('Unnamed: 6', axis=1)
    congestion_ = congestion_.drop('Unnamed: 12', axis=1)
    congestion_ = congestion_.drop('Unnamed: 26', axis=1)
    congestion_ = congestion_.drop('Unnamed: 31', axis=1)
    congestion_ = congestion_.drop('Unnamed: 35', axis=1)
    congestion_ = congestion_.drop('Unnamed: 33', axis=1)
    congestion_ = congestion_.drop('Unnamed: 37', axis=1)
    congestion_col = congestion_.iloc[1]
    congestion_.columns = congestion_col
    congestion_.drop(0, inplace=True, axis=0)
    congestion_.drop(1, inplace=True, axis=0)
    congestion_['site_id'] = congestion_['   Site   '].apply(lambda x: x.split('_')[0])
    return congestion_


def create_congestion_to_db(congestion_init, congestion_pred):
    """
    Creates congestion to database for push

    Parameters
    ----------
    congestion_init: pd.DataFrame
    congestion_pred: pd.DataFrame

    Returns
    -------

    """
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    band_by_site = site[['site_id', 'region', 'cell_band']].drop_duplicates() \
        .groupby(['site_id', 'region'])['cell_band'].apply(list).reset_index()
    band_by_site = band_by_site.rename(columns={'cell_band': 'existed_band'})
    congestion_pred['is_congested'] = np.where(
        congestion_pred[conf['COLNAMES']['CELL_SPACE']].isin(
            list(congestion_init[congestion_init[
                                     'Cell Congested'] is True][conf['COLNAMES']['CELL_SPACE']])),
        1, np.where(congestion_pred['Cell Congested'] is True, 2, 0))
    congestion_pred['date'] = np.where(congestion_pred['is_congested'] == 2,
                 conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE'],
                 np.where(congestion_pred['is_congested'] == 1,
                          "2023-02-05", 'No congestion'))

    congestion_pred = congestion_pred.rename(
        columns={conf['COLNAMES']['CELL_SPACE']: 'cell_name', 'latitude': 'site_latitude',
                 'longitude': 'site_longitude',
                 'Sector Configuration': 'bands_upgraded'})
    congestion_pred = congestion_pred.merge(band_by_site, on=['site_id', 'region'])

    congestion_pred['congestion'] = 'TDD'
    congestion_pred['runtime'] = ''
    congestion_pred['site_tech'] = '4G'
    congestion_pred['tech_upgraded'] = 'No upgrade'
    columns_to_select = ['site_id', 'site_geotype', 'site_tech', 'date', 'week_period',
                         'congestion', 'bands_upgraded', 'tech_upgraded', 'region', 'site_latitude',
                         'site_longitude', 'is_congested', 'existed_bands', 'cell_name', 'cell_band'
        , 'is_rural', 'is_ville_tdd', 'max_dl_powerload', 'max_prb_load_dl',
                         'max_throughput_user_dl_kbps']
    len(columns_to_select)
    return congestion_pred


def push_forecast_to_db():
    """
    Push forecast to database

    Returns
    -------

    :noindex:
    """
    df_predicted_traffic_kpis = pd.read_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'traffic_weekly_predicted_kpis.csv')
        , sep='|')
    df_traffic_weekly_kpis = pd.read_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                                      'traffic_weekly_kpis.csv'), sep='|')
    max_date_histo = df_traffic_weekly_kpis.date.max()
    df_predicted_traffic_kpis = (
        df_predicted_traffic_kpis)[df_predicted_traffic_kpis.date > max_date_histo]
    #
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    df_predicted_traffic_kpis_ = prepare_forecast_to_db(df_predicted_traffic_kpis, site)
    result = pipeline_format_data_to_db(df_predicted_traffic_kpis_)

    # Push Forecast
    result.rename(columns={'id': 'site', 'type_period': 'periode_type', 'period': 'periode'},
                  inplace=True)
    result['traffic_data_band'] = ''
    result.to_csv(os.path.join(conf['PATH']['FINAL'], "forecast_to_db_final.csv"), sep="|",
                  index=False)
    result = pd.read_csv(os.path.join(conf['PATH']['FINAL'], "forecast_to_db_final.csv"), sep="|")
    write_sql_database(result, "df_tdd_predicted_traffic_kpis_agg")


def push_congestion_to_db():
    """
    Push congestion to Database

    Returns
    -------

    """
    congestion_init = pd.read_excel(os.path.join(conf['PATH']['RANDIM'],
                                                 'congestion_initiale.xlsx'), engine='openpyxl',
                                    sheet_name=0)
    congestion_after = pd.read_excel(os.path.join(conf['PATH']['RANDIM'],
                                                  'congestion_forecasted.xlsx'), engine='openpyxl',
                                     sheet_name=0)
    df_traffic_weekly_kpis = pd.read_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                                      'traffic_weekly_kpis.csv'), sep='|')

    max_date_histo = df_traffic_weekly_kpis.date.max()
    congestion_init = change_randim_congestion(congestion_init)
    congestion_pred = change_randim_congestion(congestion_after)
    # Create features for congestion in db
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    site['is_ville_tdd'] = np.where(site.cell_name.str.contains('TDD'), 1, 0)
    site_ville = site.groupby('ville').agg({'is_ville_tdd': 'max'}).reset_index()
    site = site.drop('is_ville_tdd', axis=1).merge(site_ville, on='ville', how='left')
    band_by_site = site[['site_id', 'region', 'cell_band']].drop_duplicates() \
        .groupby(['site_id', 'region'])['cell_band'].apply(list).reset_index()
    band_by_site = band_by_site.rename(columns={'cell_band': 'existed_bands'})
    congestion_pred['is_congested'] = np.where(
        congestion_pred[conf['COLNAMES']['CELL_SPACE']].isin(
            list(congestion_init
                 [congestion_init['Cell Congested'] is True][conf['COLNAMES']['CELL_SPACE']])),
        1, np.where(congestion_pred['Cell Congested'] is True, 2, 0))

    d = conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE']
    r = datetime.strptime(d + '-1', "%Y%W-%w")
    congestion_date_pred = r.strftime("%Y-%m-%d")
    congestion_yearweek_pred = d
    congestion_yearweek_init = datetime.strptime(
        max_date_histo, "%Y-%m-%d").strftime("%Y%W")
    congestion_pred['date'] = np.where(congestion_pred['is_congested'] == 2, congestion_date_pred,
                                       np.where(congestion_pred['is_congested'] == 1,
                                                max_date_histo, ''))
    congestion_pred['week_period'] = np.where(congestion_pred['is_congested'] == 2,
                                              congestion_yearweek_pred,
                                              np.where(congestion_pred['is_congested'] == 1,
                                                       congestion_yearweek_init, ''))
    congestion_pred = congestion_pred.rename(columns={conf['COLNAMES']['CELL_SPACE']: 'cell_name',
                                                      'latitude': 'site_latitude',
                                                      'longitude': 'site_longitude',
                                                      'Frequency Band': 'bands_upgraded'})
    congestion_pred = congestion_pred.merge(band_by_site, on=['site_id'])
    congestion_pred = congestion_pred.merge(site, on=['site_id', 'region', 'cell_name'])
    congestion_pred = congestion_pred.rename(columns={'Environment': 'site_geotype'})
    congestion_pred['is_rural'] = np.where(congestion_pred.site_geotype == 'rural', 1, 0)
    # Merge with predicted data to have kpi needed
    congestion_pred['congestion'] = 'TDD'
    congestion_pred['site_tech'] = '4G'
    congestion_pred['tech_upgraded'] = 'No upgrade'
    columns_to_select = ['site_id', 'site_geotype', 'site_tech', 'date', 'week_period',
                         'congestion', 'bands_upgraded', 'tech_upgraded', 'region', 'site_latitude',
                         'site_longitude', 'is_congested', 'existed_bands', 'cell_name',
                         'cell_band',
                         'is_rural', 'is_ville_tdd']
    congestion_pred = congestion_pred.rename(columns={'longitude': 'site_longitude',
                                                      'latitude': 'site_latitude'})
    congestion_result = congestion_pred[columns_to_select]
    congestion_result = congestion_result.sort_values(['site_id', 'cell_name', 'is_congested'],
                                                      ascending=False)
    congestion_result.to_csv(os.path.join(conf['PATH']['FINAL'], "congestion_to_db_final.csv"),
                             sep="|", index=False)
    congestion_result = pd.read_csv(os.path.join(conf['PATH']['FINAL'],
                                                 "congestion_to_db_final.csv"), sep="|")
    write_sql_database(congestion_result, "tdd_sites_congestion_status")


def load_df_for_get_lat_long_congest_cells():
    """
    Function that read files needed to push densification congested cells to DB

    Returns
    -------
    df_randim_output_densif : densification output file from RANDim
    df_sites : typologie file generated for RANDim densification request

    """
    df_randim_output_densif = pd.read_excel(os.path.join(conf['PATH']['RANDIM'],
                                                         'densification_result.xlsx'),
                                            engine='openpyxl', sheet_name='Densification',
                                            skiprows=[1])
    df_sites = pd.read_excel(os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx'),
                             engine='openpyxl')
    return df_randim_output_densif, df_sites


def get_lat_long_congest_cells():
    """
    Add Latitude and Longitude of congest cells in the randim output file
    Add new column (congested_cells)

    Parameters:
    -----------
    df_randim_output_densif: pd.DataFrame
        Output randim dataframe
    df_sites: pd.DataFrame
        dictionnary of sites with latitude and longitude

    Returns
    -------
    densification_site: pd.DataFrame
        Randim output with latitude and longitude for conget cells
    """
    df_randim_output_densif, df_sites = load_df_for_get_lat_long_congest_cells()
    # Get set of all congest cells
    list_congest_cell = set(df_randim_output_densif['Congested Cells'].str.split(',').explode())

    # temp df with lat and long of cells
    df_cell_locations = (
        df_sites)[df_sites['cell_name'].isin(list_congest_cell)][['cell_name', 'Y_latitude',
                                                                  'X_longitude']]

    # rename
    df_cell_locations.columns = ['cell_name', 'latitude', 'longitude']

    df_randim_output_densif['cell_locations'] = df_randim_output_densif['Congested Cells'].apply(
        lambda x: [row.to_dict() for _, row in df_cell_locations[
            df_cell_locations['cell_name'].isin(x.split(','))]
        .iterrows()])
    df_randim_output_densif.drop(columns=df_randim_output_densif.columns[0], axis=1, inplace=True)
    densification_site = df_randim_output_densif
    # densification_site['cell_locations'] = densification_site["cell_locations"].apply(
    #     lambda x: json.dumps(x))
    densification_site['cell_locations'] = densification_site["cell_locations"].apply(json.dumps)

    densification_site.to_csv(os.path.join(conf['PATH']['FINAL'], "densification_site.csv"),
                              sep="|", index=False)
    return densification_site


def push_densifications_site_to_db():
    """
    Take the randim output with lat and long of congest cells and
    write it in the Database

    Parameters
    ----------
    Return
    ------
    (Call the write_sql_databse function)
    None: boolean
    """
    densification_site = get_lat_long_congest_cells()
    densification_site.rename(columns={'Polygon Number': 'polygon_number',
                                       'Cells': 'densification_sites',
                                       'Congested Cells': 'congested_cells',
                                       'Offloading Carriers': 'offloading_carriers',
                                       'Location Latitude': 'latitude',
                                       'Location Longitude': 'longitude',
                                       'cell_locations': 'cells_location'}, inplace=True)
    densification_site = densification_site.reindex(columns=['polygon_number',
                                                             'densification_sites',
                                                             'congested_cells',
                                                             'offloading_carriers',
                                                             'cells_location',
                                                             'longitude', 'latitude'])

    write_sql_database(densification_site, "densification_site")
