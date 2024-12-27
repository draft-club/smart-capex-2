import configparser
import logging
import os
from collections import namedtuple

import pandas as pd

import src.d02_preprocessing.OMA.preprocessing_hourly_oss as pho
import src.d02_preprocessing.OMA.read_process_oss_counter as rpoc
import src.d02_preprocessing.conversion_rate as cr
import src.d03_processing.OMA.economical_modules.gross_margin_quantification as gmq
import src.d03_processing.OMA.economical_modules.npv_computation as npv
import src.d03_processing.OMA.output_to_bd as db
import src.d03_processing.OMA.technical_modules.apply_traffic_gain_densification as adm
import src.d03_processing.OMA.technical_modules.traffic_by_region as tbr
import src.d03_processing.OMA.technical_modules.train_densification_impact_model as tdm
from src.d03_processing.OMA.new_site_modules.apply_new_site import (apply_new_site_data,
                                                                    apply_new_site_voice)
from src.d03_processing.OMA.new_site_modules.train_new_site import (train_new_site_data,
                                                                    train_new_site_voice)

from src.d00_conf.conf import conf
from src.d01_utils.utils import get_last_folder, super_decorator
from src.d03_processing.OMA.technical_modules.arpu_quantification import compute_revenues_per_site,\
    compute_increase_of_arpu_by_the_upgrade,compute_revenues_per_site_site
from src.d04_randim.call_randim import ApiRandim


TMP_SUFFIX_CSV = '_from_capacity.csv'
TMP_SUFFIX_XLSX = '_from_capacity.xlsx'

config = configparser.ConfigParser()
config.read('running_config.ini')
print(config.sections())
logger = logging.getLogger(__name__)


@super_decorator
def preprocessing_pipeline(tech, pod):
    """
    Run the preprocessing functions for raw hourly oss counters and weekly oss counters

    Parameters
    ----------
    tech: str
        "4G" or "TDD"
    pod: boolean
        True to Preprocess On Demand for RANDim
    pr: boolean
        True to Preprocess Regular for SC module

    Returns
    -------
    houlry_oss_preprocessed: pd.DataFrame
        a Dataframe with the corresponding format for RANDim template builder
    traffic_weekly_kpis: pd.Dataframe
        the traffic_weekly_kpis used for the rest of the process

    """
    logging.info("Start Preprocessing weekly for %s", conf['USE_CASE'])
    # Read processed data from capacity
    df_processed = pd.read_csv(
        os.path.join(conf['PATH']['CAPACITY'], 'processed_oss_all.csv'), sep='|')
    # df_processed = df_processed[df_processed['site_id'].isin(conf["SITE_TO_FILTER_TEST_NEW"])]
    logging.info("Shape of Input Processed DataFrame: %s", df_processed.shape)

    # Separate TDD and FDD data
    traffic_weekly_kpis = rpoc.preprocess_oss_weekly_from_capacity(df_processed,
                                                                   conf["PREPROCESSING"][
                                                                       "CELL_FILTER"],
                                                                   conf['USE_CASE'])

    if pod:
        print("Start preprocessing on demand")
        hourly_oss_preprocessed = pho.preprocessing_file_all(tech=tech)

        # Filter on cell name who are in traffic_weekly_kpis
        hourly_oss_preprocessed = hourly_oss_preprocessed[
            hourly_oss_preprocessed.CELL.isin(list(traffic_weekly_kpis.cell_name.unique()))]
        hourly_oss_preprocessed.to_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                    "preprocessed_template_builder_all_" + conf[
                                                        'USE_CASE'] + ".csv"),
                                       index=False, sep="|")
    else:
        print("Reading hourly data preprocessed")
        hourly_oss_preprocessed = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                           "preprocessed_template_builder_all_")
                                              + conf['USE_CASE'] + ".csv", sep="|")
    traffic_weekly_kpis.to_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'processed_oss_') + conf['USE_CASE']
        + '.csv', index=False, sep='|')
    print("End preprocessing")
    logging.info("Shape of Output Traffic Weekly KPIs: %s", traffic_weekly_kpis.shape)
    logging.info("Shape of Output Preprocessed template builder all: %s",
                 hourly_oss_preprocessed.shape)
    return traffic_weekly_kpis, hourly_oss_preprocessed


@super_decorator
def forecast_pipeline():
    """
    Run the forecasting functions on the traffic weekly kpi and return weekly_kpi and
    predicted_weekly_kpis

    Returns
    -------
    traffic_weekly_kpis: pd.DataFrame
        Dataset containing weekly KPI after forecast processing
    predicted_traffic_kpis: pd.DataFrame
        Dataset containing the forecasted weekly KPI
    """
    print("Forecasting by Prophet")
    traffic_weekly_kpis = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                   'processed_oss_') + conf['USE_CASE'] + '.csv',
                                      sep='|')
    predicted_traffic_kpis = pd.read_csv(
        os.path.join(conf['PATH']['CAPACITY'], 'df_predicted_weekly_kpis.csv'),
        sep='|')

    logging.info("Shape of Input Traffic Weekly KPIs: %s", traffic_weekly_kpis.shape)
    logging.info(
        "Shape of Input Predicted Traffic Kpi: %s", predicted_traffic_kpis.shape)

    predicted_traffic_kpis = rpoc.preprocess_oss_weekly_from_capacity(predicted_traffic_kpis,
                                                                      conf["PREPROCESSING"][
                                                                          "CELL_FILTER"],
                                                                      conf['USE_CASE'])
    traffic_weekly_kpis.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], "traffic_weekly_kpis_" + conf['USE_CASE'] +
                     ".csv"), sep='|', index=False)
    predicted_traffic_kpis.to_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], "traffic_weekly_predicted_kpis_") + conf[
            'USE_CASE'] + ".csv",
        sep='|', index=False)
    logging.info("Shape of Output Traffic Weekly KPIs: %s", traffic_weekly_kpis.shape)
    logging.info(
        "Shape of Output Predicted Traffic Weekly KPIs: %s",
        predicted_traffic_kpis.shape)
    return traffic_weekly_kpis, predicted_traffic_kpis

@super_decorator
def conversion_rate_pipeline(compute_rate, compute_export_to_randim):
    """
    Run the conversion rate functions to create and apply the rate. The results are saved in
    conf['PATH']['RANDIM']

    Parameters
    ----------
    compute_rate: bool
        Compute the conversion rate
    compute_export_to_randim: bool
        Compute the LTE transformation with the conversion rate
    Return
    ------
    conversion_rate: pd.Dataframe
        Conversion rate for each cell
    LTE: pd.Dataframe
        RANDim input transformed with the rate
    """
    if compute_rate:
        data_forecasted_path = (
                os.path.join(conf['PATH']['MODELS_OUTPUT'], "traffic_weekly_predicted_kpis_")
                + conf['USE_CASE'] + ".csv")
        data_template_builder_path = (os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                   "preprocessed_template_builder_all_")
                                      + conf['USE_CASE'] + ".csv")
        data_historical_path = (
                os.path.join(conf['PATH']['MODELS_OUTPUT'], "traffic_weekly_kpis_") +
                conf['USE_CASE'] + ".csv")
        forecasted_data = pd.read_csv(data_forecasted_path, sep='|')
        counters_data = pd.read_csv(data_historical_path, sep='|')
        template_builder = pd.read_csv(data_template_builder_path, sep='|')
        logging.info("Shape of Input Forecast Data: %s", forecasted_data.shape)
        logging.info("Shape of Input Counter Data: %s", counters_data.shape)
        logging.info("Shape of Template Builder: %s", template_builder.shape)

        conversion_rate = cr.compute_conversion_rate(
            predict_yearweek=conf['TRAFFIC_FORECASTING']['WEEK_OF_THE_UPGRADE'],
            forecast_data=forecasted_data,
            counter_data=counters_data,
            input_template_builder=template_builder)
        path_cr = os.path.join(conf['PATH']['RANDIM'], "conversion_rate_") + conf[
            'USE_CASE'] + ".csv"
        conversion_rate.to_csv(path_cr, sep='|', index=False)
    if compute_export_to_randim:
        path_cr = os.path.join(conf['PATH']['RANDIM'], "conversion_rate_") + conf[
            'USE_CASE'] + ".csv"
        print("Adapt RANDim output with conversion rate")

        path_lte = (os.path.join(conf['PATH']['RANDIM'],
                                 "LTE_preprocessed_template_builder_all_init_") + conf['USE_CASE']
                    + ".xlsx")
        lte = cr.change_lte_forecasted(conversion_rate_path=path_cr, lte_input_path=path_lte,
                                       use_case=conf['USE_CASE'])
        lte.to_csv(
            os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_') + conf['USE_CASE'] + '.csv'
            , sep='|')
        lte.to_excel(os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_' + conf['USE_CASE'] +
                                  '.xlsx'), index=False, sheet_name='LTE')

        # to insert in fucntion
        cr.post_process_lte(os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_') + conf[
            'USE_CASE'] + '.xlsx',
                            os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_') + conf[
                                'USE_CASE'] + '.xlsx')
        logging.info("Shape of Output LTE: %s", lte.shape)


@super_decorator
def prepare_densification_topology_file():
    """
    Run the pipeline to create the topoplogy file for RANDim density. It will be
    saved in the RANDim directory
    """
    print("Create topology file for RANDim densification module")
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    logging.info("Shape of Input Site: %s", site.shape)
    topology_randim = site[['site_id', 'cell_name', 'cell_id', 'latitude', 'longitude', 'azimuth']]
    topology_randim.columns = ['Site', 'cell_name', 'cell_id', 'Y_latitude', 'X_longitude',
                               'azimuth']
    topology_randim.to_excel(os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx'),
                             index=False)
    logging.info("Shape of Output Topology Randim: %s", topology_randim.shape)
    return topology_randim


@super_decorator
def train_densification_model_pipeline():
    """
    Run the pipeline to push to train the model of impact of density action
    """
    # Read Input Files
    kpis = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'processed_oss_') + conf[
        'USE_CASE'] + '.csv', sep='|')
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    site_densif = pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'], 'sites_MES.xlsx'),
                                engine='openpyxl')

    if conf['USE_CASE'] == 'TDD':
        site_densif = pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'],
                                                 'Deployment History VF.xlsx'),
                                    engine='openpyxl', sheet_name='TDD')

        # site_densif['Activation date'] = pd.to_datetime(site_densif['Activation date'])
        site_densif['year'] = site_densif['Activation date'].dt.year
        site_densif = site_densif.rename(columns={'Site name ': 'Site'})
        site_densif = site_densif[['year', 'Site']]
    # Logging info of Input files
    logging.info("Shape of Input KPIs: %s", kpis.shape)
    logging.info("Shape of Input Site:  %s", site.shape)
    logging.info("Shape of Input Deployment Densif:  %s", site_densif.shape)

    site = tdm.compute_date_deployment(kpis, site, site_densif)

    site_to_model, new_sites, kpi_before_after = tdm.pipeline_train_model_newsite_deployment(kpis,
                                                                                             site)
    # Logging info of ouput files
    logging.info("Shape of Output site to model: %s", site_to_model.shape)
    logging.info("Shape of Output New Site: %s", new_sites.shape)
    logging.info("Shape of Output KPI Before After: %s", kpi_before_after.shape)

    # Function will first group the OSS at regional granularity and will then a linear model
    # at regional level
    tbr.train_regional_model(kpis)

    # Add Code Paul train data
    if conf['DIRECTORIES']['DATA_DIRECTORY'] == 'data/samples':
        print('Skip')
    else:
        train_new_site_data()
        train_new_site_voice()



@super_decorator
def get_randim_densification_result():
    """
    This function use api randim to get differents resut
        - Call get_congestion_forecasted to get congestion forecasted file.
        - Call get_result_randim to get result densification result file (we will use
        this file in apply pipeline)

    Returns
    -------

    """
    # Initialise Randim API
    randim_api = ApiRandim()

    # Compute LTE / Congestion
    path_forecasted_file = os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_') + conf[
        'USE_CASE'] + '.xlsx'
    path_config_file = os.path.join(conf['PATH']['RANDIM'], 'run_OMA', conf['USE_CASE'],
                                    'template_builder', 'config_file_') + conf['USE_CASE'] + '.json'
    path_output_file_congestion_forecasted = os.path.join(conf['PATH']['RANDIM'],
                                                          'congestion_forecasted') + conf[
                                                 'USE_CASE'] + TMP_SUFFIX_XLSX
    info_compute_congestion = randim_api.get_congestion_forecasted(
        path_forecasted_file,
        path_config_file,
        path_output_file_congestion_forecasted)
    logging.info(info_compute_congestion)

    # Compute Densification / Randim reuslt
    path_parameters_densif = os.path.join(conf['PATH']['RANDIM'], 'densification',
                                          'parametersInput.json')
    path_congestion_forecasted = os.path.join(conf['PATH']['RANDIM'], 'congestion_forecasted') + \
                                 conf['USE_CASE'] + TMP_SUFFIX_XLSX
    path_topology_randim = os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx')
    path_output_file_densif = os.path.join(conf['PATH']['RANDIM'], 'densification',
                                           'densification_result_') + conf[
                                  'USE_CASE'] + TMP_SUFFIX_XLSX

    info_result_densification = randim_api.get_result_randim(path_parameters_densif,
                                                             path_congestion_forecasted,
                                                             path_topology_randim,
                                                             path_output_file_densif)
    print(info_result_densification)


def load_input_files():
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    randim_file = pd.read_excel(
        os.path.join(conf['PATH']['RANDIM'], 'densification_result_') + conf['USE_CASE'] +
        '_from_capacity.xlsx',
        engine='openpyxl',
        sheet_name='Densification'
    )
    pred = pd.read_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'traffic_weekly_predicted_kpis_') +
        conf["USE_CASE"] + '.csv',
        sep='|'
    )

    logging.info("Shape of Input Site: %s", site.shape)
    logging.info("Shape of Input Randim File: %s", randim_file.shape)
    logging.info("Shape of Input Traffic weekly predicted kpi: %s", pred.shape)

    return site, randim_file, pred
@super_decorator
def apply_densification_model_pipeline():
    """
    Run the pipeline to push to apply the density model to new sites (from RANDim)
    """
    # Read Input Files
    #site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']),
    # sep='|')
    #randim_file = pd.read_excel(os.path.join(conf['PATH']['RANDIM'], 'densification_result_') +
    #                            conf['USE_CASE'] + '_from_capacity.xlsx', engine='openpyxl',
    #                            sheet_name='Densification')
    #pred = pd.read_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
    # 'traffic_weekly_predicted_kpis_')
    #                   + conf["USE_CASE"] + '.csv', sep='|')
#
    ## Logging Info Input Files
    #logging.info("Shape of Input Site: %s", site.shape)
    #logging.info("Shape of Input Randim File: %s", randim_file.shape)
    #logging.info("Shape of Input Traffic weekly predicted kpi: %s", pred.shape)
    site, randim_file, pred = load_input_files()

    # Function to apply the traffic gain densification
    site_to_model, prediction, _ = adm.pipeline_apply_traffic_gain_densification(
        randim_file,
        pred,
        site)

    if conf['DIRECTORIES']['DATA_DIRECTORY'] == 'data/samples':
        print('Skip')
    else:
        prediction_data_coverage = apply_new_site_data()
        prediction_voice_coverage = apply_new_site_voice()



    prediction_data_coverage = pd.read_csv(
        os.path.join(conf['PATH']['RAW_DATA'], 'traffic_prediction_density_fdd.csv'),sep=',')
    prediction_voice_coverage = pd.read_csv(
        os.path.join(conf['PATH']['RAW_DATA'], 'traffic_prediction_density_voice.csv'),sep=',')

    #if conf["USE_CASE"] == 'TDD':
    #    train_tdd()
    #    site_to_model_coverage, prediction_coverage = predict_tdd()

    KPIPerSite = namedtuple('KPIPerSite',
                            'prediction, kpi, neighboor')

    kpi_per_sites = [KPIPerSite(prediction, 'data', False),
                     KPIPerSite(prediction_data_coverage, 'data', True),
                     KPIPerSite(prediction, 'voice', False),
                     KPIPerSite(prediction_voice_coverage, 'voice', True)]
    list_df = []
    for kpi_per_site in kpi_per_sites:
        result = adm.compute_traffic_improvement(site_to_model, kpi_per_site.prediction,
                                                 kpi=kpi_per_site.kpi, site=kpi_per_site.neighboor)
        list_df.append(result)
    df_predicted_increase_in_traffic_by_densifcation = list_df[0]
    df_predicted_increase_in_traffic_by_densifcation_site = list_df[1]
    df_predicted_increase_in_traffic_by_densifcation_voice = list_df[2]
    df_predicted_increase_in_traffic_by_densifcation_voice_site = list_df[3]

    #df_predicted_increase_in_traffic_by_densifcation_site_groupby = (
    #    df_predicted_increase_in_traffic_by_densifcation_site.groupby('site'))
    # Data
    params_for_predict_improvment = {
        'model_path': conf["PATH"]["MODELS"],
        'output_route': conf["PATH"]["MODELS_OUTPUT"],
        'variable_to_group_by': "site_region",
        'max_yearly_increment': conf["REGIONAL_TREND"]["MAX_YEARLY_INCREMENT"],
        'kpi_to_compute_trend': "total_data_traffic_dl_gb",
        'max_weeks_to_consider_increase': conf["REGIONAL_TREND"]["MAX_WEEKS_TO_CONSIDER_INCREASE"],
        'min_weeks_to_consider_increase': conf["REGIONAL_TREND"]["MIN_WEEKS_TO_CONSIDER_INCREASE"],
        'weeks_to_wait_after_the_upgrade': conf["REGIONAL_TREND"]["WEEKS_TO_WAIT_AFTER_UPGRADE"]
    }
    df_predicted_increase_in_traffic_by_densifcation_regional = (
        tbr.predict_improvement_traffic_trend_kpis(
            df_predicted_increase_in_traffic_by_densifcation, site,
            params_for_predict_improvment
        ))

    df_predicted_increase_in_traffic_by_densifcation_regional_site = (
        tbr.predict_improvement_traffic_trend_kpis(
            df_predicted_increase_in_traffic_by_densifcation_site, site,
            params_for_predict_improvment
        ))
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    rate = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], 'df_traffic_data_rate_by_year.csv',
                                    ),sep=',')
    df_cluster_key = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                              'df_cluster_keys_' + conf[
                                                  'USE_CASE'] + TMP_SUFFIX_CSV), sep='|')
    df_predicted_increase_in_traffic_by_densifcation_regional_site = (
        tbr.apply_rate_for_new_site(rate=rate,
                                    df=df_predicted_increase_in_traffic_by_densifcation_site,
                                    cluster_key=df_cluster_key,
                                    site=site))
    df_predicted_increase_in_traffic_by_densifcation_regional_site['year'] = (
        df_predicted_increase_in_traffic_by_densifcation_regional_site['year'].astype('object'))

    df_predicted_increase_in_traffic_by_densifcation_regional_site.rename(
        columns={'trafic_improvement_with_rate': 'traffic_increase_due_to_the_upgrade'},
        inplace=True)


    df_predicted_increase_in_traffic_by_densifcation_regional_split = tbr.split_data_traffic(
        df_predicted_increase_in_traffic_by_densifcation_regional)

    df_predicted_increase_in_traffic_by_densifcation_regional_split_site = tbr.split_data_traffic(
        df_predicted_increase_in_traffic_by_densifcation_regional_site)

    # Voice
    params_for_predict_improvment_voice = {
        'model_path': conf["PATH"]["MODELS"],
        'output_route': conf["PATH"]["MODELS_OUTPUT"],
        'variable_to_group_by': "site_region",
        'max_yearly_increment': conf["REGIONAL_TREND"]["MAX_YEARLY_INCREMENT"],
        'kpi_to_compute_trend': "total_voice_traffic_kerlangs",
        'max_weeks_to_consider_increase': conf["REGIONAL_TREND"]["MAX_WEEKS_TO_CONSIDER_INCREASE"],
        'min_weeks_to_consider_increase': conf["REGIONAL_TREND"]["MIN_WEEKS_TO_CONSIDER_INCREASE"],
        'weeks_to_wait_after_the_upgrade': conf["REGIONAL_TREND"]["WEEKS_TO_WAIT_AFTER_UPGRADE"]
    }
    df_predicted_increase_in_traffic_by_densifcation_regional_voice = (
        tbr.predict_improvement_traffic_trend_kpis(
            df_predicted_increase_in_traffic_by_densifcation_voice, site,
            params_for_predict_improvment_voice
        ))

    df_predicted_increase_in_traffic_by_densifcation_regional_voice_site = (
        tbr.predict_improvement_traffic_trend_kpis(
            df_predicted_increase_in_traffic_by_densifcation_voice_site, site,
            params_for_predict_improvment_voice
        ))

    #NEW
    site = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']), sep='|')
    rate_voice = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'],
                                          'df_traffic_voice_rate_by_year.csv',), sep=',')
    df_cluster_key = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                              'df_cluster_keys_' + conf[
                                                  'USE_CASE'] + TMP_SUFFIX_CSV), sep='|')
    df_predicted_increase_in_traffic_by_densifcation_regional_voice_site = (
        tbr.apply_rate_for_new_site_voice(rate=rate_voice,
                                    df=df_predicted_increase_in_traffic_by_densifcation_voice_site,
                                    cluster_key=df_cluster_key,
                                    site=site))

    df_predicted_increase_in_traffic_by_densifcation_regional_voice_site['year'] = (
        df_predicted_increase_in_traffic_by_densifcation_regional_voice_site['year'].
        astype('object'))

    df_predicted_increase_in_traffic_by_densifcation_regional_voice_site.rename(
        columns={'trafic_improvement_with_rate': 'traffic_increase_due_to_the_upgrade'},
        inplace=True)

    df_cluster_key = adm.create_cluster_from_site_to_model(site_to_model)
    df_cluster_key.to_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'df_cluster_keys_') +
                          conf['USE_CASE'] + TMP_SUFFIX_CSV, sep='|', index=False)

    logging.info("Shape of Output Site To Model: %s", site_to_model.shape)
    logging.info("Shape of Output Prediction: %s", prediction.shape)
    logging.info(
        "Shape of Output df_predicted_increase_in_traffic_by_densifcation: %s",
        df_predicted_increase_in_traffic_by_densifcation.shape)
    logging.info(
        "Shape of Output df_predicted_increase_in_traffic_by_densifcation_regional: %s",
        df_predicted_increase_in_traffic_by_densifcation_regional.shape)
    logging.info(
        "Shape of Output df_predicted_increase_in_traffic_by_densifcation_regional_split: %s",
        df_predicted_increase_in_traffic_by_densifcation_regional_split.shape)
    print(df_predicted_increase_in_traffic_by_densifcation_regional_voice)

    df_predicted_increase_in_traffic_by_densifcation_regional_split[
        'traffic_increase_due_to_the_upgrade_voice'] = (
        df_predicted_increase_in_traffic_by_densifcation_regional_voice[
            'traffic_increase_due_to_the_upgrade'])

    df_predicted_increase_in_traffic_by_densifcation_regional_split_site[
        'traffic_increase_due_to_the_upgrade_voice'] = (
        df_predicted_increase_in_traffic_by_densifcation_regional_voice_site[
            'traffic_increase_due_to_the_upgrade'])

    print(df_predicted_increase_in_traffic_by_densifcation_regional_split)
    df_predicted_increase_in_traffic_by_densifcation_regional_split.to_csv(
        os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                     "increase_in_traffic_due_to_the_upgrade_splitted", conf["EXEC_TIME"],
                     "df_predicted_increase_in_traffic_due_to_the_upgrade_spllited_" + conf[
                         "USE_CASE"] + "_from_capacity.csv"), sep="|", index=False)

    df_predicted_increase_in_traffic_by_densifcation_regional_split_site.to_csv(
        os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                     "increase_in_traffic_due_to_the_upgrade_splitted", conf["EXEC_TIME"],
                     "df_predicted_increase_in_traffic_due_to_the_upgrade_spllited_site_" + conf[
                         "USE_CASE"] + "_from_capacity.csv"), sep="|", index=False)


@super_decorator
def densification_to_economical_pipeline():
    """
    Function to moving from the technical part and technical kpi to the economic
    part and kpi

    - process/read Customer Portfolio
    - Cross CDR data and Customer portfolio
    - Train unit price model
    - Predict trafic unit price per site
    - Compute ARPU increase due to traffic increase

    Returns
    -------
    revenues_per_unit_traffic: pd.DataFrame
        Dataset with information and kpi of historical data (more especially on neighbor)
    df_increase_arpu_due_to_the_upgrade: pd.DataFrame
        Dataset with information and kpi of predicted data (more especially on neighbor)
    """
    # Read all Dataframes
    df_opex = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'df_cout_interco_site_year.csv'), sep='|')
    df_traffic_weekly_kpis = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                      'processed_oss_' + conf[
                                                          'USE_CASE'] + '.csv'),
                                         sep='|')
    df_unit_prices = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'df_unit_prices.csv'), sep='|')
    df_weights = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'df_traffic_box_mobile_year.csv'), sep=';',
        decimal=',')
    df_cluster_key = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                              'df_cluster_keys_' + conf[
                                                  'USE_CASE'] + TMP_SUFFIX_CSV), sep='|')
    df_sites = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'],
                                        conf['FILE_NAMES']['SITES']), sep='|', usecols=['site_id',
                                                                                        'region'])
    post_paid_pre_paid = pd.read_csv(os.path.join(conf['PATH']['RAW_DATA'],
                                                  'post_paid_pre_paid_region.csv'), sep=';')

    prediction_data_coverage = pd.read_csv(
        os.path.join(conf['PATH']['RAW_DATA'], 'traffic_prediction_density_fdd.csv'),sep=',')
    prediction_voice_coverage = pd.read_csv(
        os.path.join(conf['PATH']['RAW_DATA'], 'traffic_prediction_density_voice.csv'),sep=',')

    logging.info("Shape of Input Opex: %s", df_opex.shape)
    logging.info("Shape of Input Traffic Weekly Kpis: %s", df_traffic_weekly_kpis.shape)
    logging.info("Shape of Input Unit Price: %s", df_unit_prices.shape)
    logging.info("Shape of Input Weight: %s", df_weights.shape)
    logging.info("Shape of Input Cluster Key: %s", df_cluster_key.shape)
    # Compute df_revenues_per_site
    df_revenues_per_site, df_post_pre_paid_per_region = compute_revenues_per_site(
        df_traffic_weekly_kpis,
        df_opex,
        df_unit_prices,
        df_weights,
        df_cluster_key,
        df_sites,
        post_paid_pre_paid)



    df_revenues_per_site_site = compute_revenues_per_site_site(
        prediction_data_coverage,
        prediction_voice_coverage,
        df_opex,
        df_unit_prices,
        df_weights,
        df_cluster_key,
        df_sites,
        post_paid_pre_paid)

    print(df_revenues_per_site_site)
    # TMP FOR PAUL FILE
    # df_revenues_per_site = compute_revenues_per_site_paul_file(df_traffic_weekly_kpis,
    #                                                 df_opex,
    #                                                 df_unit_prices,
    #                                                 df_weights)

    latest_folder = get_last_folder(os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                                                 'increase_in_traffic_due_to_the_upgrade_splitted'))
    print(latest_folder)
    df_predicted_increase_in_traffic_by_the_upgrade = pd.read_csv(
        os.path.join(get_last_folder(os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                                                'increase_in_traffic_due_to_the_upgrade_splitted')),
                     'df_predicted_increase_in_traffic_due_to_the_upgrade_spllited_') +
        conf['USE_CASE'] + '_from_capacity.csv', sep='|'
    )

    df_predicted_increase_in_traffic_by_the_upgrade_site = pd.read_csv(
        os.path.join(get_last_folder(os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                                                  'increase_in_traffic_due_to_the_upgrade_splitted')
                                     ),
                     'df_predicted_increase_in_traffic_due_to_the_upgrade_spllited_site_') +
        conf['USE_CASE'] + '_from_capacity.csv', sep='|'
    )

    revenues_per_unit_traffic = df_revenues_per_site.copy()

    # Compute df_incrase_arpu_due_to_the_upgrade
    df_increase_arpu_due_to_the_upgrade = (
        compute_increase_of_arpu_by_the_upgrade(df_predicted_increase_in_traffic_by_the_upgrade,
                                                df_unit_prices, df_post_pre_paid_per_region))

    df_increase_arpu_due_to_the_upgrade_site = (
        compute_increase_of_arpu_by_the_upgrade(
            df_predicted_increase_in_traffic_by_the_upgrade_site,
            df_unit_prices, df_post_pre_paid_per_region))

    # Save results
    revenues_per_unit_traffic.to_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'revenues_per_unit_traffic_' + conf["USE_CASE"] + TMP_SUFFIX_CSV),
        sep="|", index=False)
    df_increase_arpu_due_to_the_upgrade.to_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_' + conf[
                         "USE_CASE"] + TMP_SUFFIX_CSV), sep="|", index=False)

    df_increase_arpu_due_to_the_upgrade_site.to_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_site_' + conf[
                         "USE_CASE"] + TMP_SUFFIX_CSV), sep="|", index=False)

    logging.info("Shape of Output revenues_per_unit_traffic: %s",
                 revenues_per_unit_traffic.shape)
    logging.info("Shape of Output df_increase_arpu_due_to_the_upgrade: %s",
                 df_increase_arpu_due_to_the_upgrade.shape)
    return revenues_per_unit_traffic, df_increase_arpu_due_to_the_upgrade


@super_decorator
def density_economical_pipeline():
    """
    - Compute Margin per site
    - Compute the margin increase
    - Compute the increase in Cash Flow
    - Compute increase  in NPV

    Returns
    -------
    df_npv: pd.DataFrame
        Final result of density module
    """
    revenues_per_unit_traffic = pd.read_csv(os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                                                         'tech_to_eco',
                                                         'revenues_per_unit_traffic_') + conf[
                                                "USE_CASE"] + TMP_SUFFIX_CSV,
                                            sep='|')
    df_increase_arpu_due_to_the_upgrade = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                     'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_') + conf[
            "USE_CASE"] + TMP_SUFFIX_CSV, sep='|')

    df_increase_arpu_due_to_the_upgrade_site = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'],
                     'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_site_') + conf[
            "USE_CASE"] + TMP_SUFFIX_CSV, sep='|')

    logging.info("Shape of Input revenues_per_unit_traffic: %s",
                 revenues_per_unit_traffic.shape)
    logging.info("Shape of Input df_increase_arpu_due_to_the_upgrade: %s",
                 df_increase_arpu_due_to_the_upgrade.shape)
    print('starting economical part')
    # ## Compute the margin per site separating both data and voice traffic
    df_margin_per_site = gmq.compute_site_margin(revenues_per_unit_traffic)

    df_increase_in_margin_due_to_the_upgrade = gmq.compute_increase_of_yearly_site_margin(
        revenues_per_unit_traffic,
        df_increase_arpu_due_to_the_upgrade=df_increase_arpu_due_to_the_upgrade,
        df_margin_per_site=df_margin_per_site)


    df_increase_in_margin_due_to_the_upgrade_site = gmq.compute_increase_of_yearly_site_margin(
        revenues_per_unit_traffic,
        df_increase_arpu_due_to_the_upgrade=df_increase_arpu_due_to_the_upgrade_site,
        df_margin_per_site=df_margin_per_site)

    logging.info(df_increase_in_margin_due_to_the_upgrade_site)

    # Melt add site + neighbour
    df_increase_in_margin_due_to_the_upgrade['increase_yearly_margin_due_to_the_upgrade'] = ((
        df_increase_in_margin_due_to_the_upgrade)['increase_yearly_margin_due_to_the_upgrade'] +
        df_increase_in_margin_due_to_the_upgrade_site['increase_yearly_margin_due_to_the_upgrade'])



    # Transform the increase in margin in increase in cash flow with the opex and capex costs
    df_increase_cash_flow_due_to_the_upgrade = npv.compute_increase_cash_flow(
        df_increase_in_margin_due_to_the_upgrade)

    # Transform the increase in cash flow in increase in NVP
    df_npv = npv.compute_npv(df_increase_cash_flow_due_to_the_upgrade)
    logging.info("Shape of Outpt NPV: %s", revenues_per_unit_traffic.shape)

    return df_npv


@super_decorator
def push_to_db_pipeline(forecast, congestion, densification_congested_cell):
    """
    Run the pipeline to push to db the forecast and the congestion

    Parameters
    ----------
    forecast: bool
        Compute the forecast to db
    congestion: bool
        Compute the congestion to db
    densification_congested_cell: bool
        Compute the densification_congested_cell

    Return
    ------
    """
    if forecast:
        print("Push to DB forecast")
        db.push_forecast_to_db()
    if congestion:
        print("Push to DB congestion")
        db.push_congestion_to_db()
    if densification_congested_cell:
        print("Push to DB densification congested cells")
        db.push_densifications_site_to_db()
