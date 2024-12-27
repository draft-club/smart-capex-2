import argparse
from ssl import create_default_context
import logging
from sys import path as pylib
import os
import re
import pandas as pd


pylib.append(os.path.dirname(os.path.realpath(__file__))+'/../..')

from config.elasticsearch_config import Elasticsearch_config
from elasticsearch_service import ElasticsearchService
from datetime import datetime
from datetime import date
import tensorflow_data_validation as tfdv
from elasticsearch_dsl import Q


_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

class MissingEnvironmentVariable(Exception):
    pass

class ImportElasticsearchConfiguration(Exception):
    pass

def validDateType(argDateStr):
    """custom argparse *date* type for user dates values given from the command line"""
    try:
        return datetime.strptime(argDateStr, "%Y-%m-%d")
    except ValueError:
        msg = "Given Date ({0}) not valid! Expected format, YYYY-MM-DD!".format(argDateStr)
        raise argparse.ArgumentTypeError(msg)

if __name__ == '__main__':
    logging.basicConfig(format=_FORMAT, level=logging.INFO)
    _logger = logging.getLogger(__name__)
    _logger.info(f'Start processing')    
    
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", help="defined platform conf, if not present : default conf is taken",
                        choices=['elkaas_bots', 'elkaas_bots_new', 'elkaas_bots_dq', 'elkaas_bots_dq_new', 'local'],
                        default='local')
    parser.add_argument("--weekstarting", help="start date (YYYY-MM-DD)", type=validDateType, default=date.today())
    parser.add_argument("--elkaaspwd", help="password for ELKaaS user only")
    parser.add_argument("--certfile", help="cerfile location if required",
                        default='/etc/ssl/certs/Orange_Internal_G2_Root_CA.pem')
    parser.add_argument("--countrylist", help="comma or space delimited list of country codes",
                        type=lambda s: re.split(',| ', s))

    args = parser.parse_args()
    _logger.info('args')
    _logger.info(args)
    # setting environment variables
    if not args.platform == 'local':
        if args.elkaaspwd:
            os.environ['ELKAAS_M2M8578_PASSWORD'] = args.elkaaspwd
        else:
            if not 'ELKAAS_M2M8578_PASSWORD' in os.environ:
                    _logger.error('Missing password for ELKaaS user')
                    raise MissingEnvironmentVariable("ELKAAS_M2M8578_PASSWORD environment variable does not exist")
                    

    # Set Elasticsearch connection
    extra = {}
    if not args.platform == 'local':
        context = create_default_context(cafile=args.certfile)
        extra = {'scheme': Elasticsearch_config.config()['elasticsearch'][args.platform]['scheme'],
                 'http_auth_username': Elasticsearch_config.config()['elasticsearch'][args.platform]['user'],
                 'http_auth_password': Elasticsearch_config.config()['elasticsearch'][args.platform]['password'],
                 'ssl_context': context}

    datasourceELK = ElasticsearchService(Elasticsearch_config.config()['elasticsearch'][args.platform]['host'],
                                         Elasticsearch_config.config()['elasticsearch'][args.platform]['port'],
                                         **extra)
    
    # Function to retrieve one week of data

    def getKafkaDataDayFromCountry(date, country_code):
        index = 'kafka-bothub-*'
        date_range_query = {'df_date': {'gte': date, 'lte': date}}
        query = Q("range", **date_range_query) & ~Q("term", resp__event_name__keyword='NEW_DIALOG_SESSION') & Q("term", ctxt__country__keyword=country_code.lower())
        return datasourceELK.get_documents_with_q(index, query)

    def getKafkaDataWeekCFromCountry(start_date, country_code):
        weekly_result = pd.DataFrame()
        date_range = pd.date_range(start=start_date, periods=7, closed=None)
        for single_date in date_range:
            _logger.info('retrieving data of ' + str(single_date)[:10] + ' for ' + country_code)
            daily_results = getKafkaDataDayFromCountry(str(single_date)[:10], country_code)
            _logger.info('got ' + str(daily_results.shape[0]) + ' documents')
            weekly_result = pd.concat([weekly_result, daily_results], ignore_index=True)
        return weekly_result

    # function for data cleaning (just keep essential data)

    def clean_results_for_data_quality(result_df):
        fields_to_keep = ['df_date', 'df_country', 'df_bot_name', 'ctxt.bot_id', 'resp.nlu_intent_name',
                          'resp.understood_nlu', 'resp.understood', 'resp.input_type', 'ctxt.country']
        
        if result_df.shape[0] == 0:
            df = pd.DataFrame(columns=fields_to_keep)
            return df 
        
        value_df = result_df['ctxt'].apply(pd.Series)
        value_df = value_df.add_prefix('ctxt.')
        value_df = pd.concat([result_df.drop(columns=['ctxt'], axis=1), value_df], axis=1)
        resp_value_df = value_df['resp'].apply(pd.Series)
        resp_value_df = resp_value_df.add_prefix('resp.')
        value_df = pd.concat([value_df.drop(columns=['resp'], axis=1), resp_value_df], axis=1)
        return value_df[fields_to_keep]

    # data collection
    # et the end of the process, we have 2 dicts: one with data from the "current week", one with data from the previous
    # week only fields from fields_to_keep
    # each dict has a first key which is the country and a second: the bot_id

    reference_data_dict = {}
    analyzed_data_dict = {}
    for country_code in args.countrylist:
        reference_data_dict[country_code] = {}
        analyzed_data_dict[country_code] = {}
        dates = pd.date_range(start=args.weekstarting, periods=2, freq='-7D')
        reference_data = clean_results_for_data_quality(getKafkaDataWeekCFromCountry(dates[1], country_code)).copy()
        analyzed_data = clean_results_for_data_quality(getKafkaDataWeekCFromCountry(dates[0], country_code)).copy()
        bot_ids = analyzed_data['ctxt.bot_id'].unique()

        for bot_id in bot_ids:
            reference_data_dict[country_code][bot_id] = reference_data[reference_data['ctxt.bot_id'] == bot_id]
            analyzed_data_dict[country_code][bot_id] = analyzed_data[analyzed_data['ctxt.bot_id'] == bot_id]


    # changing the type of value.understood and value.understood_nlu from int64 to bytes
    # otherwise we won't get drifts
    for country_code in args.countrylist:
        for bot_id in analyzed_data_dict[country_code]:
            reference_data_dict[country_code][bot_id] = reference_data_dict[country_code][bot_id]\
                .astype({'resp.understood_nlu': bytes, 'resp.understood': bytes})
            analyzed_data_dict[country_code][bot_id] = analyzed_data_dict[country_code][bot_id]\
                .astype({'resp.understood_nlu': bytes, 'resp.understood': bytes})

    # validate the last week for each country and each bot 
    _logger.info('Validation')
    for country_code in args.countrylist:
        for bot_id in analyzed_data_dict[country_code] :
           
            # Ignore test bots
            if bot_id == '8bd44fd7146b2ee00b8d15b8e48a4ae8':
                continue

            input_types = ['message::text', 'message::postback']

            reference_df = reference_data_dict[country_code][bot_id]
            analyzed_df = analyzed_data_dict[country_code][bot_id]

            # skip the weeks when there's no data in the analyzed df or reference one
            if reference_df.shape[0] == 0 or analyzed_df.shape[0] == 0:
                continue

            # for text and postback
            for input_type in input_types:

                _logger.info('country : {0}  -  botid : {1}'.format(country_code, bot_id))
                _logger.info('input type : {0}'.format(input_type))

                input_reference_df = reference_df[reference_df['resp.input_type'] == input_type]
                input_analyzed_df = analyzed_df[analyzed_df['resp.input_type'] == input_type]

                # skip the weeks when there's no data in the filtered dataframe
                if input_reference_df.shape[0] == 0 or input_analyzed_df.shape[0] == 0:
                    continue

                """
                the validations will be in two steps : 
                    - validate the received data frame entirely. the result is a dataframe we named anomalies_df
                    - validate only the rows with a true value.understood, then we will change the anomaly description
                     of value.intent in anomalies_df with the new description that we got validating the filtered 
                     dataframe 
                """

                analyzed_fields = ['resp.nlu_intent_name', 'resp.understood', 'resp.understood_nlu']

                # validation of understood & understood_nlu :
                reference_stats = tfdv.generate_statistics_from_dataframe(input_reference_df)
                analyzed_stats = tfdv.generate_statistics_from_dataframe(input_analyzed_df)

                # Inferring a schema from reference batch
                schema = tfdv.infer_schema(reference_stats, max_string_domain_size=999999999)
                tfdv.get_feature(schema, 'resp.nlu_intent_name').drift_comparator.infinity_norm.threshold = 0
                tfdv.get_feature(schema, 'resp.understood')     .drift_comparator.infinity_norm.threshold = 0
                tfdv.get_feature(schema, 'resp.understood_nlu') .drift_comparator.infinity_norm.threshold = 0
                tfdv.get_feature(schema, 'resp.understood')     .distribution_constraints.min_domain_mass = 0
                tfdv.get_feature(schema, 'resp.understood_nlu') .distribution_constraints.min_domain_mass = 0

                anomalies = tfdv.validate_statistics(schema=schema,
                                                     previous_statistics=reference_stats,statistics=analyzed_stats)

                anomalies_df = tfdv.utils.display_util.get_anomalies_dataframe(anomalies)

                # drift detection in value.intent : understood == true
                understood_reference_df = input_reference_df[input_reference_df['resp.understood'] == b'True']
                understood_analyzed_df = input_analyzed_df[input_analyzed_df['resp.understood'] == b'True']

                # do not check intent if there is no line with understand == true
                if understood_reference_df.shape[0] != 0 and understood_analyzed_df.shape[0] != 0:

                    reference_stats = tfdv.generate_statistics_from_dataframe(understood_reference_df)
                    analyzed_stats = tfdv.generate_statistics_from_dataframe(understood_analyzed_df)

                    anomalies = tfdv.validate_statistics(schema=schema,
                                                         previous_statistics=reference_stats,
                                                         statistics=analyzed_stats)

                    intent_anomaly = tfdv.utils.display_util.get_anomalies_dataframe(anomalies)
                    if not intent_anomaly.empty:
                        new_anomaly_row = intent_anomaly.loc[["'\\'{0}\\''".format(analyzed_fields[0])], :]
                        anomalies_df.loc[["'\\'{0}\\''".format(analyzed_fields[0])], :] = new_anomaly_row

                for field in analyzed_fields:
                    try:
                        field_anomaly_df = anomalies_df.loc[["'\\'{0}\\''".format(field)]].copy()
                        description = field_anomaly_df.iloc[0, 1]
                    except KeyError:
                        description = None

                    id = bot_id+'_'+str(dates[0])+'_'+field+'_'+input_type
                    field_anomaly_df['_id']              = id
                    field_anomaly_df['df_bot_id']        = bot_id
                    field_anomaly_df['df_bot_name']      = analyzed_data_dict[country_code][bot_id].iloc[0, 2]
                    field_anomaly_df['df_country_code']  = country_code
                    field_anomaly_df['df_country']       = analyzed_data_dict[country_code][bot_id].iloc[0, 1]
                    field_anomaly_df['df_date']          = str(dates[0])
                    field_anomaly_df['dq_field']         = "{0} - {1}".format(field, input_type)

                    """
                    when there is new values & drift, the description text is similar to :
                    Examples contain values missing from the schema: Examples contain values missing from the schema:
                     'ADSL not stable' (<1%), 'ION service' (~2%) . The Linfty distance between current and previous is
                      0.0607272 (up to six significant digits), above the threshold 0. The feature value with maximum 
                      difference is: 'ايموجي 😉'
                    
                    this string has to be processed to :
                       - know whether there is new values or not, this might be done by looking for the :
                        'Examples contain values missing'

                       - get the drift value : the string contained between 'is' and '(' 
                         & the value with max drift : the sting after the ':'

                       - get the new values if there is some : the string after :
                         'Examples contain values missing from the schema' by spliting the list using '), ' as a 
                         separator. for each element in the resulted array we will try to remove the frequency to be 
                         pushed as the dq_new_values field the values with their frequencies are in the 
                         dq_new_values_frequencies field 
   
                    """

                    # if sep > -1 then the anomaly text contains two anomalies : new values and drift
                    sep = description.find('The Linfty distance between current') if description else -1

                    # the description string will be processed to get the needed information
                    if sep != -1:

                        domain_beg = description.find('Examples contain values missing from the schema')
                        domain_end = description.find('%). ', domain_beg) if domain_beg > -1 else None
                        domain = description[domain_beg + 49: domain_end + 2] if domain_beg > -1 else ''
                        drift = description[sep + 3:]

                        field_anomaly_df['dq_drift'] = float(
                            drift[drift.find('is ') + 3:drift.find('(')]) if drift else 0
                        field_anomaly_df['dq_value'] = drift[drift.find(':') + 2:] if drift else ''

                        new_values = domain.split('), ') if domain else []
                        dq_new_values_frequency = []

                        for i in range(len(new_values)):
                            element = new_values[i]
                            element = element if element[-1] == ')' else element + ')'
                            dq_new_values_frequency += [element]
                            end = re.findall('\([<,~]\d+%\)$', element)
                            end_index = element.find(end[-1])
                            new_values[i] = element[:end_index - 1]

                        field_anomaly_df['dq_new_values'] = None
                        field_anomaly_df.at["'\\'{0}\\''".format(field), 'dq_new_values'] = new_values
                        field_anomaly_df['dq_new_values_count'] = len(new_values)
                        field_anomaly_df['dq_new_values_frequencies'] = None
                        field_anomaly_df.at["'\\'{0}\\''".format(field),
                                            'dq_new_values_frequencies'] = dq_new_values_frequency

                    else:
                        field_anomaly_df['dq_drift'] = float(description[description.find('is ')+3:
                                                                         description.find('(')]) if description else 0
                        field_anomaly_df['dq_value'] = description[description.find(':')+2:] if description else ''

                    if description:
                        field_anomaly_df.drop('Anomaly short description', axis=1, inplace=True)
                        field_anomaly_df.drop('Anomaly long description', axis=1, inplace=True)
                        field_anomaly_df.reset_index(drop=True, inplace=True)

                    dict = field_anomaly_df.to_dict('records')
                    saved_doc_number = datasourceELK.parallel_import_documents('kafka-botanalytics-connector-bot-weekly-drifts', dict)

    # Closing connection to ES
    datasourceELK.get_client().transport.close() 
    
    _logger.info(f'End of processing')             
            
