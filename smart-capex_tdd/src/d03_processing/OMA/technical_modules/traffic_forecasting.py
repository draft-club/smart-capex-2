"""Traffic Forecasting"""
import os
import random
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from prophet import Prophet

from src.d00_conf.conf import conf
from src.d01_utils.utils import check_path, add_logging_info
from src.d01_utils.utils import write_csv_sql



# Function using in prophet to filter data inferior to 2
def is_valid_cell(df):
    """
    The is_valid_cell function checks if a DataFrame has more than two non-null values in the
    "traffic_kpis" column.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing a column named "traffic_kpis"

    Returns
    -------
    A boolean value indicating whether the DataFrame has more than two non-null values in the
    "traffic_kpis" column.

    """
    return df["traffic_kpis"].notna().sum() > 2


def get_traffic_kpis_data(df_traffic_weekly_kpis_):
    """
    Function that gets all the data needed from training the traffic kpis

    Parameters
    ----------
    df_traffic_weekly_kpis_ : pd.DataFrame
        Data from preprocessing containing weekly OSS counter

    Returns
    -------
    df_traffic_weekly_kpis : pd.DataFrame
        prepared dataframe for forecasting

    """
    # choose the right column
    my_columns = ["cell_name", "cell_tech", "cell_band", "site_id", 'cell_sector', "date", "year",
                 "week", "week_period"]
    my_columns += conf['TRAFFIC_KPI']["TRAFFIC_4G"]

    df_traffic_weekly_kpis = df_traffic_weekly_kpis_[my_columns]
    return df_traffic_weekly_kpis


@add_logging_info
def use_prophet(df_traffic_weekly_kpis_):
    """
    Function that apply fbprophet model on the data using paralelisation.
    First run a function to prepare data. Then run the model to have predicted value

    Parameters
    ----------
    df_traffic_weekly_kpis_: pd.DataFrame

    Returns
    -------
    df_predicted_traffic_kpis: pd.DataFrame
        forecasted data
    df_traffic_weekly_kpis: pd.DataFrame
        weekly kpis used to forecast

    """

    df_traffic_weekly_kpis = get_traffic_kpis_data(df_traffic_weekly_kpis_)
    df_traffic_weekly_kpis, df_predicted_traffic_kpis = prediction(df_traffic_weekly_kpis)
    return df_traffic_weekly_kpis, df_predicted_traffic_kpis


def performance_metrics(df_predicted_traffic_kpis, df_traffic_weekly_kpis,
                        #output_file=os.path.join(conf['PATH']['MODELS_OUTPUT'],
                        #                         'Traffic_forecasting', "prophet",
                        #                         conf['EXEC_TIME']),
                        output_file='fake_rate'):
    """
    Function calculating the performance of the model

    Parameters
    ----------
    df_predicted_traffic_kpis: pd.DataFrame
        Forecast data
    df_traffic_weekly_kpis: pd.DataFrame
        Historical data
    output_file: str
        path to save the model
    Returns
    -------
    errors_prediction: pd.DataFrame
        errors_prediction dataset
    """
    # Reformat file
    # df_traffic_weekly_kpis['date'] = df_traffic_weekly_kpis['date'].apply(
    # lambda x: datetime.strftime(x, '%Y-%m-%d'))
    df_predicted_traffic_kpis['ds'] = df_predicted_traffic_kpis['ds'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.rename(columns={'ds': 'date'})
    # Merge test and prediction
    # Create metrics
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.reset_index(drop=True)
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.merge(df_traffic_weekly_kpis, how="left",
                                                                on=["cell_name", "date",
                                                                    "traffic_kpis",
                                                                    "cell_tech", "cell_band",
                                                                    "site_id",
                                                                    "cell_sector"
                                                                    ]
                                                                )
    df_predicted_traffic_kpis["AE"] = abs(
        df_predicted_traffic_kpis["yhat"] - df_predicted_traffic_kpis['traffic_kpi_values'])
    df_predicted_traffic_kpis["SAPE"] = df_predicted_traffic_kpis["AE"] / (
            df_predicted_traffic_kpis[
                "traffic_kpi_values"] + df_predicted_traffic_kpis["yhat"])
    errors_prediction = df_predicted_traffic_kpis.groupby(["cell_name", "traffic_kpis"]).agg(
        {"AE": "mean",
         "SAPE": "mean",
         "traffic_kpi_values": "sum",
         "yhat": "sum"})
    errors_prediction = errors_prediction.reset_index()
    errors_prediction = errors_prediction.rename(
        columns={'SAPE': 'SMAPE', 'yhat': 'sum_pred', 'AE': 'MAE',
                 'traffic_kpi_values': 'sum_values'})
    write_csv_sql(df=df_predicted_traffic_kpis, name='prediction_errors_cells.csv',
                  path=output_file,
                  separator=conf['CSV']['SEP'], sql=False, csv=True, if_exists='replace')
    errors_prediction['SMAPE'] = errors_prediction['SMAPE'].apply(lambda x: float(x) * 2)
    for _, j in enumerate(list(errors_prediction.traffic_kpis.unique())):
        error = errors_prediction[errors_prediction.traffic_kpis == j]
        err = []
        err.append(error[error.SMAPE <= 0.05].shape[0])
        err.append(error[(error.SMAPE > 0.05) & (error.SMAPE <= 0.1)].shape[0])
        err.append(error[(error.SMAPE > 0.1) & (error.SMAPE <= 0.2)].shape[0])
        err.append(error[(error.SMAPE > 0.2) & (error.SMAPE <= 0.3)].shape[0])
        err.append(error[(error.SMAPE > 0.3) & (error.SMAPE <= 0.4)].shape[0])
        err.append(error[(error.SMAPE > 0.4)].shape[0])
        count_cell_smape = pd.DataFrame(err, index=['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
                                                    '>40%'], columns=['Number_of_cells'])
        count_cell_smape = count_cell_smape.reset_index()
        count_cell_smape.columns = ['Interval', 'Number_of_cells']
        write_csv_sql(df=count_cell_smape, name='count_cell_SMAPE_' + str(j) + '.csv',
                      path=output_file,
                      separator=conf['CSV']['SEP'], sql=False, csv=True, if_exists='replace')
    return errors_prediction


def prepare_train(df_traffic_weekly_kpis):
    """
    Functions that prepare for the prophet model

    Parameters
    ----------
    df_traffic_weekly_kpis: pd.DataFrame)
        dataframe containing the traffic weekly kpis

    Return
    ------
    df_traffic_weekly_kpis: pd.DataFrame
        dataset containing the weekly kpis prepared for training
    """
    # Compute lag that will be useful in the case of cross_validation
    df_traffic_weekly_kpis['date'] = pd.to_datetime(df_traffic_weekly_kpis['date'],
                                                    format="%Y-%m-%d")
    min_date = df_traffic_weekly_kpis["date"].min()
    df_traffic_weekly_kpis["lag"] = ((df_traffic_weekly_kpis["date"] - min_date)
                                     //
                                     pd.Timedelta(1, 'W')) + 1
    # Prepare data to use it in prophet: long data format
    id_vars = ["cell_name", "cell_sector", "date", "cell_tech", "cell_band", "site_id", "lag"]
    df = pd.melt(df_traffic_weekly_kpis, id_vars=id_vars,
                 value_vars=conf["TRAFFIC_KPI"]["TRAFFIC_4G"])

    df.columns = id_vars + ["traffic_kpis", "traffic_kpi_values"]

    def keep_kpi(row):
        if row["cell_tech"] == "4G":
            if row["traffic_kpis"] in (conf["TRAFFIC_KPI"]["TRAFFIC_4G"]):
                return 1
        return None

    df["iscorrect"] = df.apply(keep_kpi, axis=1)
    df1 = df[df["iscorrect"] == 1]
    df1 = df1.drop(["iscorrect"], axis=1)
    df1["traffic_kpi_values"] = df1.groupby(['traffic_kpis', "cell_name"])[
        "traffic_kpi_values"].apply(
        lambda x: x.fillna(x.mean()))
    df_traffic_weekly_kpis = df1.fillna(0)
    return df_traffic_weekly_kpis


def prediction(df_traffic_weekly_kpis):
    """
    Run the traffic forecasting function (based on the prophet algorithm) associated with the
    current country

    Rework site id if necessary (according to the current country)

    Parameters
    ----------
    df_traffic_weekly_kpis: pd.Dataframe
        Dataframe containing weekly KPI from preprocessing

    Return
    ------
    df_traffic_weekly_kpis: pd.DataFrame
        Dataframe containing weekly KPI from preprocessing
    df_predicted_traffic_kpis: pd.DataFrame
        Dataframe containing weekly prediction

    """
    output_file = os.path.join(conf['PATH']['MODELS_OUTPUT'],
                               'Traffic_forecasting', "prophet", conf['EXEC_TIME'])
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    print("Begin forecasting")
    df_traffic_weekly_kpis = prepare_train(df_traffic_weekly_kpis)

    # Fit and predict all forecasts model/data
    # Parallelization
    df_predicted_traffic_kpis = fit_predict_all_forecasts(df_traffic_weekly_kpis)

    # Assign values outside boundary to boundary values
    clip_predicted_traffic_kpis(df_predicted_traffic_kpis, output_file)

    # Compute performance metrics on predicted traffic_kpis
    errors_prediction = performance_metrics(df_predicted_traffic_kpis, df_traffic_weekly_kpis,
                                            output_file=output_file)

    write_csv_sql(df=errors_prediction, name="errors_prediction.csv", path=output_file,
                  separator=conf['CSV']['SEP'], sql=False,
                  csv=True, if_exists='replace')
    df_predicted_traffic_kpis, df_traffic_weekly_kpis = post_process_forecasts(
        df_predicted_traffic_kpis,
        df_traffic_weekly_kpis)
    return df_traffic_weekly_kpis, df_predicted_traffic_kpis


def clip_predicted_traffic_kpis(df_predicted_traffic_kpis, output_file):
    """
    The clip_predicted_traffic_kpis function adjusts the predicted traffic KPI values to ensure they
    fall within valid boundaries. It sets negative predictions to 0 and caps percentage-based
    predictions at 100. Additionally, it ensures the output directory exists and
    formats the date column.

    Parameters
    ----------
    df_predicted_traffic_kpis: pd.DataFrame
        DataFrame containing predicted traffic KPIs.
    output_file: str
        Path to the output directory.

    Returns
    -------
    Adjusted DataFrame with valid yhat values and formatted ds column.
    """
    is_negative = df_predicted_traffic_kpis["yhat"] < 0
    is_percent_over_100 = (df_predicted_traffic_kpis["yhat"] > 100) & (
        df_predicted_traffic_kpis["traffic_kpis"].str.contains("percentage"))
    df_predicted_traffic_kpis.loc[is_negative, "yhat"] = 0
    df_predicted_traffic_kpis.loc[is_percent_over_100, "yhat"] = 100
    check_path(output_file)
    print("inside the metrics validation")
    check_path(output_file)
    df_predicted_traffic_kpis['ds'] = df_predicted_traffic_kpis['ds'].apply(
        lambda x: datetime.strftime(x, '%Y-%m-%d'))


def post_process_forecasts(df_predicted_traffic_kpis, df_traffic_weekly_kpis):
    """
    The post_process_forecasts function processes the predicted traffic KPIs by adding a run date,
    renaming columns, pivoting the table, and reindexing columns. It also fills missing values,
    pivots the historical traffic KPIs, and filters out predictions that are not in
    the historical data range.

    Parameters
    ----------
    df_predicted_traffic_kpis: pd.DataFrame
        DataFrame containing the predicted traffic KPIs.
    df_traffic_weekly_kpis: pd.DataFrame
        DataFrame containing the historical weekly traffic KPIs.

    Returns
    -------
    df_predicted_traffic_kpis: pd.DataFrame
        Processed DataFrame of predicted traffic KPIs.
    df_traffic_weekly_kpis: pd.DataFrame
        Processed DataFrame of historical weekly traffic KPIs.

    """
    df_predicted_traffic_kpis["week_date_run"] = conf['EXEC_TIME']
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.rename(
        columns={'ds': 'date', 'yhat': 'Valeur'})
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.pivot_table(
        index=["cell_name", "date", "cell_tech", "cell_band", "site_id", "cell_sector"],
        columns='traffic_kpis')['Valeur'].reset_index()
    df_predicted_traffic_kpis["year"] = df_predicted_traffic_kpis["date"].dt.year
    df_predicted_traffic_kpis["week"] = df_predicted_traffic_kpis["date"].dt.week
    df_predicted_traffic_kpis["week_period"] = (df_predicted_traffic_kpis["date"].dt.year * 100
                                                +
                                                df_predicted_traffic_kpis["date"].dt.week)
    # get all the columns using for the next steps
    col_to_reindex = ['cell_name', 'date', 'cell_tech', 'cell_band', 'cell_sector', 'year',
                      'week', 'week_period', 'site_id'] + (conf['TRAFFIC_KPI']['TRAFFIC_4G'])
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.reindex(col_to_reindex, axis=1)
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.fillna(0)
    df_traffic_weekly_kpis = df_traffic_weekly_kpis.pivot_table(
        index=["cell_name", "date", "cell_tech", "cell_band", "site_id", "cell_sector"],
        columns='traffic_kpis')['traffic_kpi_values'].reset_index()
    print("End of forecasting")
    max_date_histo = df_traffic_weekly_kpis.date.max()
    df_predicted_traffic_kpis = df_predicted_traffic_kpis[
        df_predicted_traffic_kpis.date > max_date_histo]
    # predicted_col = df_predicted_traffic_kpis.columns
    # print("Compare forecast with historical evolution")
    # df_predicted_traffic_kpis = apply_upper_bound(df_predicted_traffic_kpis,
    # 'processed_oss_tdd.csv')
    df_predicted_traffic_kpis = df_predicted_traffic_kpis[col_to_reindex]
    return df_predicted_traffic_kpis, df_traffic_weekly_kpis


def fit_predict_all_forecasts(df_traffic_weekly_kpis):
    """
    The fit_predict_all_forecasts function applies the Prophet forecasting model to multiple subsets
    of a DataFrame in parallel, using multiprocessing to speed up the process.
    It groups the data by specific keys, filters out invalid groups, and then fits and predicts the
    traffic KPIs for each valid group.

    Parameters
    ----------
    df_traffic_weekly_kpis: pd.DataFrame
        A pandas DataFrame containing weekly traffic KPIs data.

    Returns
    -------
    df_predicted_traffic_kpis: pd.DataFrame
        A pandas DataFrame containing the predicted traffic KPIs for each valid group.
    """
    cpu_counts = cpu_count() - 4
    # p = Pool(cpu_counts)
    print('\n Fitting prophet model \n')
    groupby_keys = ["cell_name", "traffic_kpis", "cell_tech", "cell_band", "site_id"]
    traffic_groupby = df_traffic_weekly_kpis.groupby(groupby_keys)
    list_of_df = [df for keys, df in traffic_groupby if is_valid_cell(df)]
    list_of_keys = [keys for keys, df in traffic_groupby if is_valid_cell(df)]

    def fit_predict_wrapper(args):
        df, keys = args
        return fit_predict(df, keys)

    with Pool(cpu_counts) as p:
        df_predicted_traffic_kpis = pd.concat(
            objs=list(tqdm(p.imap(fit_predict_wrapper, zip(list_of_df, list_of_keys)),
                           keys=list_of_keys))
        )
    return df_predicted_traffic_kpis


# def get_series_column_and_algorithm_parameter(series,
#                                              _cross_validation=conf["TRAFFIC_FORECASTING"][
#                                                  "TEST_TRAIN"]):
#    """
#    Function to get series column and algorithm parameter from model
#
#    Parameters
#    ----------
#    series: pandas series
#
#    _cross_validation: bool
#
#    Returns
#    -------
#    algorithm: str
#        'LBFGS' or 'Newton'
#    change_points: int
#    series: pandas series
#    series_train: pandas series
#    series_test: pandas series
#    traffic_kpi: str
#    cell_name: str
#    cell_tech: str
#    cell_band: str
#    site_idcell_sector: str
#    """
#    if _cross_validation:
#        # Split train/test at 80%
#        tot_len = len(series.date)
#        percent_split = conf['TRAFFIC_FORECASTING']['TEST_TRAIN_SPLIT'] / 100
#        date_split = series.date.iat[int(tot_len * percent_split)]
#        series_test = series[(series.date >= date_split)]
#        series_train = series[(series.date < date_split)]
#        traffic_kpi = series_train["traffic_kpis"].iat[0]
#        cell_name = series_train["cell_name"].iat[0]
#        cell_tech = series_train["cell_tech"].iat[0]
#        cell_band = series_train["cell_band"].iat[0]
#        site_id = series_train["site_id"].iat[0]
#        cell_sector = series_train["cell_sector"].iat[0]
#        # Indicate number changepoints
#
#        series_train_len = len(series_train)
#        if series_train_len < 100:
#            algorithm = 'Newton'
#            change_points = series_train_len // 2
#        else:
#            algorithm = 'LBFGS'
#            change_points = series_train_len // 4
#
#    else:
#        traffic_kpi = series["traffic_kpis"].iat[0]
#        cell_name = series["cell_name"].iat[0]
#        cell_tech = series["cell_tech"].iat[0]
#        cell_band = series["cell_band"].iat[0]
#        site_id = series["site_id"].iat[0]
#        cell_sector = series["cell_sector"].iat[0]
#        # (traffic_kpi, cell_name, cell_tech, cell_band, site_id,
#        # cell_sector) = series['traffic_kpis','cell_name','cell_tech','cell_band',
#        # 'site_id', 'cell_sector'].iloc[0].values.tolist()[0:5]
#
#        # Indicate number changepoints
#        series_len = len(series)
#        if series_len < 100:
#            algorithm = 'Newton'
#            change_points = series_len // 2
#        else:
#            algorithm = 'LBFGS'
#            change_points = series_len // 4
#        series_train, series_test = None, None
#    return (algorithm, change_points, series, series_train, series_test, traffic_kpi,
#            cell_name, cell_tech, cell_band, site_id, cell_sector)
#

def get_series_column_and_algorithm_parameter(series,
                                              _cross_validation=2):
    """
    Function to get series column and algorithm parameter from model
    Parameters
    ----------
    series: pandas series
    _cross_validation: bool
    Returns
    -------
    algorithm: str
        'LBFGS' or 'Newton'
    change_points: int
    series: pandas series
    series_train: pandas series
    series_test: pandas series
    traffic_kpi: str
    cell_name: str
    cell_tech: str
    cell_band: str
    site_idcell_sector: str
    """
    # Extract common attributes
    traffic_kpi = series["traffic_kpis"].iat[0]
    cell_name = series["cell_name"].iat[0]
    cell_tech = series["cell_tech"].iat[0]
    cell_band = series["cell_band"].iat[0]
    site_id = series["site_id"].iat[0]
    cell_sector = series["cell_sector"].iat[0]

    if _cross_validation:
        # Split train/test at 80%
        tot_len = len(series.date)
        percent_split = conf['TRAFFIC_FORECASTING']['TEST_TRAIN_SPLIT'] / 100
        date_split = series.date.iat[int(tot_len * percent_split)]
        series_test = series[(series.date >= date_split)]
        series_train = series[(series.date < date_split)]
        series_len = len(series_train)
    else:
        series_len = len(series)

    # Determine algorithm and change_points
    if series_len < 100:
        algorithm = 'Newton'
        change_points = series_len // 2
    else:
        algorithm = 'LBFGS'
        change_points = series_len // 4

    # Prepare return values
    if _cross_validation:
        return (algorithm, change_points, series, series_train, series_test,
                traffic_kpi, cell_name, cell_tech, cell_band, site_id, cell_sector)

    return (algorithm, change_points, series, None, None, traffic_kpi,
            cell_name, cell_tech, cell_band, site_id, cell_sector)


#def fit_predict_to_delete(series, n_mcmc=0):
#    """
#    Apply prophet model on dataframe via multiprocessing package
#
#    Parameters
#    ----------
#    series: series
#        Data to forecast
#    n_mcmc: int
#        sampling option
#    Return
#    -------
#    forecast_prb_d: pd.DataFrame
#        Dataframe containing prediction
#    """
#    # recuperate the max of cell ds
#    test_train = conf["TRAFFIC_FORECASTING"]["TEST_TRAIN"]
#
#    (algorithm, change_points, series, series_train, series_test, traffic_kpi, cell_name,
#    cell_tech,
#     cell_band, site_id, cell_sector) = get_series_column_and_algorithm_parameter(series)
#    # Setup model
#    model = Prophet(uncertainty_samples=0,
#                    yearly_seasonality=True,
#                    weekly_seasonality=True,
#                    daily_seasonality=True,
#                    mcmc_samples=n_mcmc,
#                    seasonality_prior_scale=0.5,
#                    n_changepoints=change_points)
#    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
#    model.add_country_holidays(country_name='MA')
#
#    # Set seed
#    random.seed(123)
#    if test_train:
#        # Fit model
#        model.fit(series_train
#                  .drop(columns=["cell_name", "traffic_kpis"])
#                  .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
#                  .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
#                  algorithm=algorithm)
#        # Make prediction
#        future = model.make_future_dataframe(
#            periods=conf['TRAFFIC_FORECASTING']['MAX_DATE_TO_PREDICT'] +
#                    len(series_test), freq='W', include_history=True)
#    else:
#        model.fit(series
#                  .drop(columns=["cell_name", "traffic_kpis"])
#                  .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
#                  .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
#                  algorithm=algorithm)
#        future = model.make_future_dataframe(
#            periods=conf['TRAFFIC_FORECASTING']['MAX_DATE_TO_PREDICT'],
#            freq='W', include_history=True)
#    forecast = model.predict(future)
#    forecast["traffic_kpis"] = traffic_kpi
#    forecast["cell_name"] = cell_name
#    forecast["cell_tech"] = cell_tech
#    forecast["cell_band"] = cell_band
#    forecast["site_id"] = site_id
#    forecast['cell_sector'] = cell_sector
#    forecast = forecast[
#        ['ds', 'yhat', 'traffic_kpis', 'cell_name', 'cell_tech', 'cell_band', 'site_id',
#         'cell_sector']]
#    return forecast


def fit_predict(series, n_mcmc=0):
    """
    Apply prophet model on dataframe via multiprocessing package

    Parameters
    ----------
    series: series
        Data to forecast
    n_mcmc: int
        sampling option
    Return
    -------
    forecast_prb_d: pd.DataFrame
        Dataframe containing prediction
    """
    # recuperate the max of cell ds
    test_train = conf["TRAFFIC_FORECASTING"]["TEST_TRAIN"]
    _cross_validation = conf["TRAFFIC_FORECASTING"][
        "TEST_TRAIN"]

    (algorithm, change_points, series, series_train, series_test, traffic_kpi, cell_name, cell_tech,
     cell_band, site_id, cell_sector) = (
        get_series_column_and_algorithm_parameter(series,_cross_validation))

    # Setup model
    model = Prophet(uncertainty_samples=0,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    mcmc_samples=n_mcmc,
                    seasonality_prior_scale=0.5,
                    n_changepoints=change_points)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_country_holidays(country_name='MA')

    # Set seed
    random.seed(123)

    # Choose the data to fit based on test_train flag
    if test_train:
        data_to_fit = series_train
    else:
        data_to_fit = series

    # Fit model
    model.fit(data_to_fit
              .drop(columns=["cell_name", "traffic_kpis"])
              .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
              .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
              algorithm=algorithm)

    # Make prediction
    future = model.make_future_dataframe(
        periods=conf['TRAFFIC_FORECASTING']['MAX_DATE_TO_PREDICT'] + len(series_test),
        freq='W', include_history=True)

    forecast = model.predict(future)

    # Add additional columns to forecast dataframe
    forecast["traffic_kpis"] = traffic_kpi
    forecast["cell_name"] = cell_name
    forecast["cell_tech"] = cell_tech
    forecast["cell_band"] = cell_band
    forecast["site_id"] = site_id
    forecast['cell_sector'] = cell_sector

    forecast = forecast[
        ['ds', 'yhat', 'traffic_kpis', 'cell_name', 'cell_tech', 'cell_band', 'site_id',
         'cell_sector']]

    return forecast
