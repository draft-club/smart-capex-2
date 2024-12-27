import logging
logger = logging.getLogger("my logger")
# from src.d02_preprocessing.OCI.read_process_sites import *
# from src.d01_utils.utils import *
from src.utils import *
from sklearn.linear_model import LinearRegression
import pickle

def compute_traffic_by_region(df_sites,
                              df_traffic_weekly_kpis,
                              kpi_to_compute_trend=['total_data_traffic_dl_gb', 'total_voice_traffic_kerlands']):
    """
    :param df_sites: site master
    :param df_traffic_weekly_kpis:
    :param kpi_to_compute_trend: variable present in the df_traffic_weekly_kpis that you want to group
    """
    df_traffic_weekly_kpis['date'] = pd.to_datetime(df_traffic_weekly_kpis['date'])
    df_sites = df_sites[['site_name', 'site_region']].drop_duplicates().groupby('site_name').first().reset_index()
    df_sites.columns = ['site_id', 'site_region']
    df_sites.site_id = df_sites['site_id'].astype(str)
    df_traffic_weekly_kpis = df_traffic_weekly_kpis.merge(df_sites,
                                                          on='site_id',
                                                          how='left')

    ## Groupby site and date and sum the kpi to compute the tred
    df = df_traffic_weekly_kpis.groupby(['site_region','date'])[kpi_to_compute_trend].sum().reset_index()
    df_list_date = list(df.date.unique())[:-1]
    df = df[df.date.isin(df_list_date)]
    return df


def train_trend_model_with_linear_regression(df_traffic_by_region,
                                             # output_route = conf['PATH']['MODELS'],
                                             variable_to_group_by = ['site_region'],
                                             kpi_to_compute_trend=['total_data_traffic_dl_gb']):

    """
    For teach region train a linear regression model to compute the trend
    :param df_traffic_by_region: dataset with the traffic by region and date
    :param output_route: route to save the model
    :param variable_to_group_by:
    :param kpi_to_compute_trend:
    :return: None
    """
    df_traffic_by_region['date'] = pd.to_datetime(df_traffic_by_region['date'])

    # Transform the dataset to easily apply a groupby afterwards
    df = pd.melt(df_traffic_by_region,
                 id_vars=["date"] + variable_to_group_by ,
                 value_vars= kpi_to_compute_trend)

    df.columns = ["date"] + variable_to_group_by + ["traffic_kpis","value_traffic_kpi"]

    def linear_regression(df, variable_to_group_by, 
                          # output_route
                         ):
        df = df.reset_index(drop=True)
        min_date = df['date'].min()
        df['difference_weeks'] = (df['date'] - min_date) / pd.Timedelta(1, 'W')
        x_train = df[["difference_weeks"]]
        y_train = df[["value_traffic_kpi"]]

        # hnote - fitting log-log model to red sea
        # if df[variable_to_group_by[0]].drop_duplicates()[0] == 'Red Sea':
        #     x_train = np.log(x_train[1:])
        #     y_train = np.log(y_train[1:])

        model = LinearRegression()
        model.fit(x_train, y_train)
        print(df[variable_to_group_by[0]].drop_duplicates()[0], " | coef : ", model.coef_, " | intercept : ", model.intercept_, " | R² : ", model.score(x_train, y_train))

        filename = (df[variable_to_group_by[0]].drop_duplicates()[0]+ ".sav")
        # trend_model_path = os.path.join(output_route,
        #                                "traffic_trend_by_" +variable_to_group_by[0],
        #                                 kpi_to_compute_trend[0],
        #                                 conf['EXEC_TIME'])

#         if not os.path.exists(trend_model_path):
#             os.makedirs(trend_model_path)

#         with open(os.path.join(trend_model_path,filename), 'wb') as f:
#             pickle.dump(model, f)
        return model

    model = df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(linear_regression,
                                                              variable_to_group_by = variable_to_group_by,
                                                              # output_route = output_route
                                                             )

    ## Train the global trend
    df = df.groupby(['date','traffic_kpis'])['value_traffic_kpi'].sum().reset_index()
    df[variable_to_group_by[0]] = "GLOBAL"
    model = df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(linear_regression,
                                                              variable_to_group_by = variable_to_group_by,
                                                              # output_route = output_route
                                                             )

    return model

# def predict_improvement_traffic_trend_kpis(df_increase_traffic_after_upgrade,
#                                            df_sites,
#                                            model_path = conf['PATH']['MODELS'],
#                                            output_route = conf['PATH']['MODELS_OUTPUT'],
#                                            variable_to_group_by = ['site_region'],
#                                            max_yearly_increment = conf['TRAFFIC_IMPROVEMENT_TREND']['MAX_YEARLY_INCREMENT'],
#                                            kpi_to_compute_trend=['total_data_traffic_dl_gb'],
#                                            max_weeks_to_predict =conf['TRAFFIC_IMPROVEMENT_TREND']['MAX_WEEKS_TO_PREDICT'],
#                                            max_weeks_to_consider_increase = conf['TRAFFIC_IMPROVEMENT_TREND']['MAX_WEEKS_TO_CONSIDER_INCREASE'],
#                                            min_weeks_to_consider_increase = conf['TRAFFIC_IMPROVEMENT_TREND']['MIN_WEEKS_TO_CONSIDER_INCREASE'],
#                                            weeks_to_wait_after_the_upgrade = conf['TRAFFIC_IMPROVEMENT']['WEEKS_TO_WAIT_AFTER_UPGRADE']):

#     ## Merge sites with region info
#     df_sites = df_sites[['site_id','site_region']].drop_duplicates().groupby('site_id').first().reset_index()
#     df_increase_traffic_after_upgrade = df_increase_traffic_after_upgrade.merge(df_sites,
#                                                                                 on = 'site_id',
#                                                                                 how = 'left')
#     ## If the site region is not available we put the global trend
#     df_increase_traffic_after_upgrade[variable_to_group_by[0]].fillna("GLOBAL", inplace = True)
#     path = os.path.join(model_path, "traffic_trend_by_" + variable_to_group_by[0], kpi_to_compute_trend[0])
#     path = os.path.join(path, getLastFile(path))

#     def predict_linear_regression(df,
#                                   model_path,
#                                   max_yearly_increment,
#                                   kpi_to_compute_trend,
#                                   max_weeks_to_predict,
#                                   max_weeks_to_consider_increase,
#                                   min_weeks_to_consider_increase,
#                                   weeks_to_wait_after_the_upgrade):
#         df = df.reset_index()
#         print(df['site_id'].drop_duplicates().iloc[0])

#         ## Load the model that contains the traffic growth in the region
#         loaded_model = pickle.load(open(os.path.join(model_path, df['site_region'].drop_duplicates()[0]+ ".sav"), 'rb'))
#         increment = loaded_model.coef_/loaded_model.intercept_[0]
#         increment = increment[0]

#         ## Set up a maximum increment per region
#         # Todo : add following values as Constant variable + explian why
#         max_weekly_increment = (max_yearly_increment/100)/52
#         if increment > max_weekly_increment:
#             increment = max_weekly_increment

#         ## Compute the lag with the week of the upgrade
#         df['week_of_the_upgrade'] = df['week_of_the_upgrade'].apply(int).apply(str)
#         df['week_period'] = df['week_period'].apply(int).apply(str)
#         df['lag_between_upgrade'] = df[['week_of_the_upgrade', 'week_period']].apply(
#             lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

#         ## Create the dataset to put the increase in traffic
#         ## Starting date: week of the upgrade
#         ## Final date: week of the upgrade + max_weeks_to_predict
#         max_date = pd.to_datetime(df.loc[df['week_of_the_upgrade'] == df['week_period']]['date'].values[0])
#         index = pd.date_range(max_date, periods=max_weeks_to_predict, freq='W')
#         df_date = pd.DataFrame({'date': index})
#         df_date['date'] = df_date['date'] + datetime.timedelta(days=1)

#         ### Average of traffic before the upgrade
#         df_before = df[(df['lag_between_upgrade'] < 0) & (df['lag_between_upgrade'] >= -8)][
#             kpi_to_compute_trend].mean()
#         df_date['traffic_before'] = df_before[0]

#         ### Increase of traffic after the upgrade
#         df_date['increase_of_traffic_after_the_upgrade'] = df['increase_of_traffic_after_the_upgrade'].drop_duplicates()[0]

#         ## The traffic that will grow is equal to the traffic before the upgrade + the increment in traffic due to the upgrade
#         #df_date['total_traffic_to_compute_increase'] = df_date['traffic_before'] + df_date['increase_of_traffic_after_the_upgrade']
#         df_date['total_traffic_to_compute_increase'] = df_date['increase_of_traffic_after_the_upgrade']
#         df_date['increase'] = df_date['increase_of_traffic_after_the_upgrade']*increment

#         df_date['week_of_the_upgrade'] = str(df['week_of_the_upgrade'].drop_duplicates()[0])
#         ## Compute date features
#         df_date['year'] = df_date['date'].apply(lambda x: str(x.year))
#         df_date['week'] = df_date['date'].apply(lambda x: str(x.week))
#         df_date['week_period'] = df_date[['year', 'week']].apply(
#             lambda x: get_week_period(x.iloc[0], x.iloc[1]), axis=1)

#         ## Compute the lag between the week of the upgrade and the week
#         df_date['lag_between_upgrade'] = df_date[['week_of_the_upgrade', 'week_period']].apply(
#             lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

#         ## After the maximum number of weeks, the increase will be 0
#         df_date['increase'] = np.where(df_date['lag_between_upgrade']> max_weeks_to_consider_increase, 0 , df_date['increase'])
#         df_date['increase'] = np.where(df_date['lag_between_upgrade']< min_weeks_to_consider_increase, 0 , df_date['increase'])
#         df_date['total_increase'] = np.cumsum(df_date['increase'])
#         df_date['traffic_increase_due_to_the_upgrade'] = df_date['total_increase']+df_date['total_traffic_to_compute_increase']   #- df_date['traffic_before']

#         ## The upgrade effect will take a minimum number of weeks to make effect
#         df_date['traffic_increase_due_to_the_upgrade'] = np.where(df_date['lag_between_upgrade']<= weeks_to_wait_after_the_upgrade, 0 , df_date['traffic_increase_due_to_the_upgrade'])
#         return df_date

#     ## For each site predicts the increase in traffic due to the upgrade
#     df_all = df_increase_traffic_after_upgrade.groupby(["site_id", "bands_upgraded"]).apply(predict_linear_regression,
#                                                                         model_path = path,
#                                                                         max_yearly_increment = max_yearly_increment,
#                                                                         kpi_to_compute_trend = kpi_to_compute_trend,
#                                                                         max_weeks_to_predict= max_weeks_to_predict,
#                                                                         max_weeks_to_consider_increase = max_weeks_to_consider_increase,
#                                                                         min_weeks_to_consider_increase = min_weeks_to_consider_increase,
#                                                                         weeks_to_wait_after_the_upgrade = weeks_to_wait_after_the_upgrade).reset_index()

#     ## Save the results in the output route
#     final_increase_traffic = os.path.join(output_route,
#                                          "increase_in_traffic_due_to_the_upgrade",
#                                           conf['EXEC_TIME'])

#     if not os.path.exists(final_increase_traffic):
#         os.makedirs(final_increase_traffic)


#     write_csv_sql(df_all, "increase_in_traffic_due_to_the_upgrade_"+kpi_to_compute_trend[0]+".csv",
#                  path=final_increase_traffic, sql=False, csv=True)
#     return df_all


# def merge_predicted_improvement_traffics(df_predicted_increase_in_traffic_data_by_the_upgrade,
#                                          df_predicted_increase_in_traffic_voice_by_the_upgrade):

#     df_predicted_increase_in_traffic_data_by_the_upgrade.columns = ['site_id', 'bands_upgraded', 'level_1', 'date', 'traffic_before_data',
#                                                                     'increase_of_traffic_after_the_upgrade_data',
#                                                                     'total_traffic_to_compute_increase_data',
#                                                                     'increase_data',
#                                                                     'week_of_the_upgrade', 'year', 'week',
#                                                                     'week_period', 'lag_between_upgrade',
#                                                                     'total_increase_data',
#                                                                     'traffic_increase_due_to_the_upgrade_data']

#     df_predicted_increase_in_traffic_voice_by_the_upgrade.columns = ['site_id', 'bands_upgraded', 'level_1', 'date', 'traffic_before_voice',
#                                                                     'increase_of_traffic_after_the_upgrade_voice',
#                                                                     'total_traffic_to_compute_increase_voice',
#                                                                     'increase_voice',
#                                                                     'week_of_the_upgrade', 'year', 'week',
#                                                                     'week_period', 'lag_between_upgrade',
#                                                                     'total_increase_voice',
#                                                                     'traffic_increase_due_to_the_upgrade_voice']

#     df_predicted_increase_in_traffic_by_the_upgrade = df_predicted_increase_in_traffic_voice_by_the_upgrade.merge(
#         df_predicted_increase_in_traffic_data_by_the_upgrade, on=['site_id', 'bands_upgraded', 'date', 'week_of_the_upgrade', 'year',
#                                                                   'week', 'week_period', 'lag_between_upgrade'],
#     how='inner')

#     df_predicted_increase_in_traffic_by_the_upgrade.to_csv(
#         os.path.join(conf['PATH']['MODELS_OUTPUT'],'increase_in_traffic_due_to_the_upgrade',
#                      conf['EXEC_TIME'],'df_predicted_increase_in_traffic_by_the_upgrade.csv'),
#         sep='|')

#     return df_predicted_increase_in_traffic_by_the_upgrade




