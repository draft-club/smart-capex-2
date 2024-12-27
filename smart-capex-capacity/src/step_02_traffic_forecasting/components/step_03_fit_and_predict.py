from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


 # pylint: disable=C0415
 # pylint: disable=R0913
@component(base_image=pipeline_config["base_image"])
def fit_and_predict(interquartile_coefficient: int,
                    cross_validation: bool,
                    train_test_split: bool,
                    training_ratio: float,
                    max_date_to_predict: int,
                    country_name: str,
                    traffic_weekly_kpis_data_input: Input[Dataset],
                    predicted_traffic_kpis_data_output: Output[Dataset]):
    """It is used to apply preprocessing steps and training with prophet based on group of keys

    Args:
        interquartile_coefficient (int): It is a coefficient used to remove the outliers as per the grouped keys
        cross_validation (bool): In case of False: It is used to perform training on the data with performing the forecast 
                                 on the dates shifted by one. In case of True: It is used to perform training on the data 
                                 with performing the forecast on the training dates
        train_test_split (bool): It is used to perform training on the data with performing the forecast 
                                 on the test data
        training_ratio (float): It is used to split the data into training and test when cross_validation = True
                                and train_test_split = True
        max_date_to_predict (int): It is used to perform training on the data with performing the forecast on next 52 days
        country_name (str): It is the configured country to get the holidays of the country with prophet
        traffic_weekly_kpis_data_input (Input[Dataset]):  It holds the processed traffic weekly KPIs
        predicted_traffic_kpis_data_output (Output[Dataset]):  It holds the predicted traffic weekly KPIs

    Returns:
        predicted_traffic_kpis_data_output: It holds the predicted traffic weekly KPIs
    """

    import random

    import multiprocess as mp
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    from scipy import stats

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    print("df_traffic_weekly_kpis shape:", df_traffic_weekly_kpis.shape)

    def get_series_column_and_algorithm_parameter(series):
        traffic_kpi = series["traffic_kpis"].iat[0]
        cell_name = series["cell_name"].iat[0]
        cell_tech = series["cell_tech"].iat[0]
        cell_band = series["cell_band"].iat[0]
        site_id = series["site_id"].iat[0]

        # Indicate number changepoints
        if len(series) < 100:
            algorithm = 'Newton'
            change_points = int(len(series) / 2)
        else:
            algorithm = 'LBFGS'
            change_points = int(len(series) / 4)

        return algorithm, change_points, series, traffic_kpi, cell_name, cell_tech, cell_band, site_id

    def correct_using_interquartile_method(series):
        q1 = series["traffic_kpi_values"].quantile(0.25)
        q3 = series["traffic_kpi_values"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - interquartile_coefficient * iqr
        upper_bound = q3 + interquartile_coefficient * iqr
        lower_outliers_index = series[series["traffic_kpi_values"] < lower_bound].index
        upper_outliers_index = series[series["traffic_kpi_values"] > upper_bound].index
        series.loc[lower_outliers_index, "traffic_kpi_values"] = series["traffic_kpi_values"].median()
        series.loc[upper_outliers_index, "traffic_kpi_values"] = series["traffic_kpi_values"].median()
        return series

    def correct_by_deleting_consecutive_zeros(series):
        l = []
        for n in np.arange(48, len(series), 2):
            last_n = series["traffic_kpi_values"].tail(n)
            num_zeros = sum(last_n == 0)
            if num_zeros < 5:
                l.append(n)
        if len(l) != 0:
            try:
                series["traffic_kpi_values"] = series["traffic_kpi_values"] \
                                                    .tail(np.max(l)) \
                                                    .replace(to_replace=0, value=np.nan) \
                                                    .fillna(method='ffill') \
                                                    .fillna(method='bfill')
                return series
            except (ValueError, TypeError, ZeroDivisionError):
                series["traffic_kpi_values"] = 9999
                return series
        else:
            series["traffic_kpi_values"] = 9999
            return series

    def get_model_and_test_set(series, algorithm, change_points):
        model = Prophet(uncertainty_samples=0,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        mcmc_samples=0,
                        seasonality_prior_scale=0.5,
                        n_changepoints=change_points)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_country_holidays(country_name=country_name)

        random.seed(123)
        max_ds = series["date"].max()
        min_ds = series["date"].min()
        if cross_validation:
            if train_test_split:
                train_max_date = series.sort_values(by="date").iloc[int(training_ratio * len(series))]["date"]
                model.fit(series[series["date"] <= train_max_date]
                          .drop(columns=["cell_name", "traffic_kpis"])
                          .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
                          .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
                          algorithm=algorithm
                          )
                x_test = pd.DataFrame(
                    pd.date_range(start=train_max_date, end=max_ds,
                                  freq='W'))  # + timedelta(days=104 * 7), freq='W'))
                x_test.columns = ["ds"]
                x_test["ds"] = x_test["ds"] + pd.DateOffset(1)
            else:
                model.fit(series
                          .drop(columns=["cell_name", "traffic_kpis"])
                          .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
                          .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
                          algorithm=algorithm
                          )
                x_test = pd.DataFrame(
                    pd.date_range(start=min_ds, end=max_ds,
                                  freq='W'))  # + timedelta(days=104 * 7), freq='W'))
                x_test.columns = ["ds"]
        else:
            model.fit(series
                      .drop(columns=["cell_name", "traffic_kpis"])
                      .rename(columns={'traffic_kpi_values': 'y', 'date': 'ds'})
                      .replace((np.abs(stats.zscore(series["traffic_kpi_values"])) > 3), None),
                      algorithm=algorithm
                      )
            x_test = pd.DataFrame(
                pd.date_range(max_ds, periods=max_date_to_predict,
                              freq='W'))
            x_test.columns = ["ds"]
        return model, x_test

    def get_forecasted_dataframe(model, x_test, traffic_kpi, cell_name, cell_tech, cell_band, site_id):
        df_forecast = model.predict(x_test)
        df_forecast["traffic_kpis"] = traffic_kpi
        df_forecast["cell_name"] = cell_name
        df_forecast["cell_tech"] = cell_tech
        df_forecast["cell_band"] = cell_band
        df_forecast["site_id"] = site_id
        return df_forecast

    def get_prophet(series):
        try:
            algorithm, change_points, series_, traffic_kpi, cell_name, cell_tech, cell_band, site_id = \
                                                                        get_series_column_and_algorithm_parameter(series)

            series_processed_1 = correct_using_interquartile_method(series_)
            series_processed_2 = correct_by_deleting_consecutive_zeros(series_processed_1)

            model, x_test = get_model_and_test_set(series_processed_2, algorithm, change_points)

            df_forecast = get_forecasted_dataframe(model, x_test, traffic_kpi, cell_name, cell_tech, cell_band, site_id)

            return df_forecast
        except (ValueError, ZeroDivisionError, TypeError,
                IOError, AttributeError, IndexError, KeyError):
            return None

    def is_valid_cell(df):
        return df["traffic_kpis"].notna().sum() > 2


    cpu_counts = 4
     # pylint: disable=E1102
    with mp.Pool(cpu_counts) as pool:
        groupby_keys = ["cell_name", "traffic_kpis", "cell_tech", "cell_band", "site_id"]
        traffic_groupby = df_traffic_weekly_kpis.groupby(groupby_keys)

        list_of_df = [df for keys, df in traffic_groupby if is_valid_cell(df)]
        list_of_keys = [keys for keys, df in traffic_groupby if is_valid_cell(df)]

        results = pool.map(get_prophet, list_of_df)

    # Concatenate results into a single DataFrame
    df_predicted_traffic_kpis = pd.concat(results, keys=list_of_keys)

    # Reset index if needed
    df_predicted_traffic_kpis.reset_index(inplace=True)

    print("df_predicted_traffic_kpis shape: ", df_predicted_traffic_kpis.shape)
    print("df_predicted_traffic_kpis columns: ", df_predicted_traffic_kpis.columns)

    df_predicted_traffic_kpis.to_parquet(predicted_traffic_kpis_data_output.path)
