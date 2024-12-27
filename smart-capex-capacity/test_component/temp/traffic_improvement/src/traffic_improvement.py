import logging
logger = logging.getLogger("my logger")
#from src.d00_conf.conf import conf, conf_loader
from src.conf import conf, conf_loader
conf_loader("OSN")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
#from src.d01_utils.utils import *
from src.utils import *
import os
import datetime
import math
import pandas as pd 
import numpy as np

## ok

def compute_traffic_model_features(df_traffic_weekly_kpis,
                                   list_of_upgrades,
                                   sites_to_remove,
                                   removes_sites_with_more_than_one_upgrade_same_cluster = True,
                                   kpi_to_compute_upgrade_effect=["total_data_traffic_dl_gb"],
                                   compute_target = True,
                                   max_neighbors = conf['TRAFFIC_IMPROVEMENT']['MAX_NUMBER_OF_NEIGHBORS'],
                                   cluster_key_separator='-'):

        df_traffic_weekly_kpis['date'] = df_traffic_weekly_kpis.week_period.apply(lambda x: datetime.datetime.strptime(str(x) + '-1', '%Y%W-%w'))

        if (removes_sites_with_more_than_one_upgrade_same_cluster):
            list_of_upgrades = list_of_upgrades[~list_of_upgrades['site_id'].isin(sites_to_remove )]

        ## Get a list with all the name of the columns depending on the maximum number of neighbors
        list_neighbors = []
        for i in range(0,max_neighbors):
            list_neighbors.append("neighbor" + "_" + str(i+1))

        df_neighbors =  pd.melt(list_of_upgrades[['cluster_key']+list_neighbors],
                                id_vars=['cluster_key'],
                                value_vars=list_neighbors)

        df_neighbors.columns = ['cluster_key',
                                'variable',
                                'neighbor_name']

        df_neighbors = df_neighbors[~(df_neighbors['neighbor_name']== "")]
        df_neighbors = df_neighbors[['neighbor_name','cluster_key']]
        df_neighbors.columns = ['site_id', 'cluster_key']

        df = pd.concat([df_neighbors, list_of_upgrades[['site_id','cluster_key']]])
        ###df.to_csv('df_feature'+kpi_to_compute_upgrade_effect[0]+'.csv')
        list_of_upgrades.rename(columns = {'selected_band':'bands_upgraded'},inplace = True)
        df = df.merge(list_of_upgrades[['cluster_key','bands_upgraded','week_of_the_upgrade']],
                      on = 'cluster_key',
                      how = 'left')


        ## Select KPI
        df_traffic_weekly_kpis_site = df_traffic_weekly_kpis.groupby(['site_id',
                                                                      'date',
                                                                      'week_period'])[kpi_to_compute_upgrade_effect].sum().reset_index()

        df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(df,
                                                                   on='site_id',
                                                                   how='left')

        ## Select KPI
        df_traffic_weekly_kpis_cluster = df_traffic_weekly_kpis_site.groupby(['cluster_key',
                                                                              'date',
                                                                              'week_period',
                                                                              'week_of_the_upgrade',
                                                                              'bands_upgraded'])[kpi_to_compute_upgrade_effect].sum().reset_index()

        def compute_traffic_features_per_cluster(df,
                                                 kpi_to_compute_upgrade_effect,
                                                 compute_target = True):

            df_final = df.reset_index(drop=True)[['cluster_key']].drop_duplicates()

            df['week_of_the_upgrade'] = df['week_of_the_upgrade'].apply(int).apply(str)
            df['week_period'] = df['week_period'].apply(int).apply(str)

            df['lag_between_upgrade'] = df[['week_of_the_upgrade', 'week_period']].apply(
                lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)


            ## Compute the model features: how was the traffic before the upgrade
            df_before = df[(df['lag_between_upgrade'] < 0) & (df['lag_between_upgrade'] >= -8)][
                kpi_to_compute_upgrade_effect].agg({'mean',
                                                    'std',
                                                    'median',
                                                    'min',
                                                    'max'}).reset_index()

            df_before = pd.pivot_table(df_before,
                                       values=kpi_to_compute_upgrade_effect,
                                       columns=['index'],
                                       aggfunc=np.sum)

            new_columns_names = []
            for i in df_before.columns:
                new_columns_names.append(kpi_to_compute_upgrade_effect[0] + "_" + i)
            df_before.columns = new_columns_names
            df_before.reset_index(drop=True, inplace=True)
            df_final = pd.concat([df_final, df_before], axis=1)

            if compute_target:
                ## Compute target variable: average of the traffic after the upgrade
                df_target = df[
                    (df['lag_between_upgrade'] > conf['TRAFFIC_IMPROVEMENT']['WEEKS_TO_WAIT_AFTER_UPGRADE']) & (
                            df['lag_between_upgrade'] <= 8)][kpi_to_compute_upgrade_effect].mean()

                df_final['target_variable'] = df_target[0]
            return df_final


        ### Compute the model
        # features
        df_traffic_features = df_traffic_weekly_kpis_cluster.groupby('cluster_key').apply(compute_traffic_features_per_cluster,
                                                                                          kpi_to_compute_upgrade_effect = kpi_to_compute_upgrade_effect,
                                                                                          compute_target = compute_target).reset_index(drop = True)
        ## save to debug
        df_traffic_features.to_csv('df_feature_+debug_'+kpi_to_compute_upgrade_effect[0]+".csv",sep='|')
        if compute_target:
            df_traffic_features = df_traffic_features[~df_traffic_features['target_variable'].isna()]

        ## Get list with all the upgrades
        return df_traffic_features


def get_capacity_kpis_features_model(list_of_upgrades,
                                     df_affected,
                                     sites_to_remove,
                                     remove_sites_with_more_than_one_upgrade_same_cluster=True,
                                     capacity_kpis_features = ["cell_occupation_dl_percentage"],
                                     operation_to_aggregate_cells = 'mean'):
    print(df_affected.columns)
    print(df_affected.head())
    df_affected['date']=df_affected.week_period.apply(lambda x: datetime.datetime.strptime(str(x) + '-1', '%Y%W-%w'))
    logger.info("Training the forecasting capacity model")
    if (remove_sites_with_more_than_one_upgrade_same_cluster):
        list_of_upgrades = list_of_upgrades[~list_of_upgrades['site_id'].isin(sites_to_remove + ["OCI0016"])]

    if operation_to_aggregate_cells == 'mean':
        df_capacity_kpis_features = df_affected.groupby(['site_id',
                                                         'date',
                                                         'week_period',
                                                         'cell_tech',
                                                         'week_of_the_upgrade',
                                                         'bands_upgraded',
                                                         'tech_upgraded'])[capacity_kpis_features].mean().reset_index()
    else:
        df_capacity_kpis_features = df_affected.groupby(['site_id',
                                                         'date',
                                                         'week_period',
                                                         'cell_tech',
                                                         'week_of_the_upgrade',
                                                         'bands_upgraded',
                                                         'tech_upgraded'])[capacity_kpis_features].sum().reset_index()

    def get_capacity_kpi_before(df, capacity_kpis_features,operation_to_aggregate_cells):

        df['week_of_the_upgrade'] = df['week_of_the_upgrade'].apply(int).apply(str)
        df['week_period'] = df['week_period'].apply(int).apply(str)

        df['lag_between_upgrade'] = df[['week_of_the_upgrade', 'week_period']].apply(
            lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

        ## Compute the model features: how was the traffic before the upgrade
        df_before = df[(df['lag_between_upgrade'] < 0) & (df['lag_between_upgrade'] >= -8)][capacity_kpis_features].agg(
            {operation_to_aggregate_cells}).reset_index()

        return df_before

    df_capacity_kpis_features = df_capacity_kpis_features.groupby(['site_id',
                                                                   'cell_tech',
                                                                   'week_of_the_upgrade',
                                                                   'bands_upgraded',
                                                                   'tech_upgraded']).apply(get_capacity_kpi_before, capacity_kpis_features = capacity_kpis_features,operation_to_aggregate_cells = operation_to_aggregate_cells).reset_index()

    df_capacity_kpis_features = df_capacity_kpis_features[['site_id',
                                                           'cell_tech']+ capacity_kpis_features]

    df = pd.pivot_table(df_capacity_kpis_features,
                           values=capacity_kpis_features,
                           index=['site_id'],
                           columns=['cell_tech'],
                           aggfunc=operation_to_aggregate_cells,
                           fill_value=None).reset_index()

    df.columns = [''.join(col).strip() for col in df.columns.values]
    return df


def compute_upgrade_typology_features(df_traffic_weekly_kpis,
                                     list_of_upgrades,
                                     sites_to_remove,
                                     remove_sites_with_more_than_one_upgrade_same_cluster=True,
                                     max_neighbors = conf['TRAFFIC_IMPROVEMENT']['MAX_NUMBER_OF_NEIGHBORS'],
                                      cluster_key_separator='-'):

    if (remove_sites_with_more_than_one_upgrade_same_cluster):
        list_of_upgrades = list_of_upgrades[~list_of_upgrades['site_id'].isin(sites_to_remove)]

    ## Get a list with all the name of the columns depending on the maximum number of neighbors
    list_neighbors = []
    for i in range(0, max_neighbors):
        list_neighbors.append("neighbor" + "_" + str(i + 1))

    df_neighbors = pd.melt(list_of_upgrades[['cluster_key','week_of_the_upgrade'] + list_neighbors],
                           id_vars=['cluster_key','week_of_the_upgrade'],
                           value_vars=list_neighbors)

    df_neighbors.columns = ['cluster_key',
                            'week_of_the_upgrade',
                            'variable',
                            'site_id']
    df_neighbors.drop(columns = ['variable'],inplace = True)
    df_neighbors = df_neighbors[~(df_neighbors['site_id'] == "")]
    df_neighbors.dropna(subset = ['site_id'], inplace = True)
    df_upgrades = pd.concat([df_neighbors, list_of_upgrades[['site_id', 'cluster_key','week_of_the_upgrade']]])


    def compute_sites_attributes(df, df_traffic_weekly_kpis, cluster_key_separator):
        df = df.reset_index()
        df_weekly = df_traffic_weekly_kpis.merge(df,
                                                 on = 'site_id',
                                                 how= 'inner')

        df_weekly = df_weekly[df_weekly['cell_band'].isin(conf['TRAFFIC_IMPROVEMENT']['BANDS_TO_CONSIDER'])]

        df_weekly['week_period'] = df_weekly['week_period'].apply(str)
        df_weekly['week_of_the_upgrade'] = df_weekly['week_of_the_upgrade'].apply(str)

        df_weekly = df_weekly[df_weekly['week_period'] < df_weekly['week_of_the_upgrade']]

        df_weekly = df_weekly[['site_id', 'cluster_key','cell_band', 'cell_tech']].drop_duplicates()
        ## Distinguish between site where the upgrade took place and other site
        # df_weekly['type'] = np.where(df_weekly['cluster_key'].apply(lambda x : x.split("_")[0]) ==df_weekly['site_id'], "site", "cluster" )
        df_weekly['type'] = np.where(df_weekly['cluster_key'].apply(lambda x: str(x).split(cluster_key_separator)[0]) == df_weekly['site_id'], "site", "cluster")

        df_number_tech = df_weekly.groupby(['cluster_key', 'cell_tech', 'type'])['site_id'].count().reset_index()

        df_number_tech = pd.pivot_table(df_number_tech,
                               values='site_id',
                               index=['cluster_key'],
                               columns=['cell_tech','type'],
                               fill_value=0,
                               aggfunc=np.sum)
        df_number_tech.columns = ['_'.join(col).strip() for col in df_number_tech.columns.values]
        df_number_tech = df_number_tech.reset_index()
        ## Number of sites on the cluster
        df_number_tech['number_of_sites_on_the_cluster'] = (df.shape[0]-1)
        return df_number_tech

    ## cheicker number of cluster
    df_upgrades_features = df_upgrades.groupby('cluster_key').apply(compute_sites_attributes,
                                                                    df_traffic_weekly_kpis=df_traffic_weekly_kpis,
                                                                    cluster_key_separator=cluster_key_separator).reset_index(drop = True)
    df_upgrades_features.fillna(0,inplace = True)
    #
    return df_upgrades_features




def get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                         df_cell_affected,
                                         list_of_upgrades,
                                         sites_to_remove,
                                         type_of_traffic = 'data',
                                         group_bands = False,
                                         remove_sites_with_more_than_one_upgrade_same_cluster=True,
                                         kpi_to_compute_upgrade_effect=["total_data_traffic_dl_gb"],
                                         capacity_kpis_features=["cell_occupation_dl_percentage"],
                                         upgraded_to_not_consider = ["2G"],
                                         compute_target_variable_traffic = True,
                                         cluster_key_separator='-'):

    df_upgrade_typology_features = compute_upgrade_typology_features(df_traffic_weekly_kpis,
                                                                     list_of_upgrades,
                                                                     sites_to_remove,
                                                                     remove_sites_with_more_than_one_upgrade_same_cluster,
                                                                     cluster_key_separator=cluster_key_separator)
    # df_upgrade_typology_features.to_csv('/data/OSN/02_intermediate/df_upgrade_typology_features.csv',
    #                                          sep='|', index=False)
    # df_upgrade_typology_features = pd.read_csv('/data/ORDC/02_intermediate/df_upgrade_typology_features.csv', sep='|')
    df_data_traffic_model_features = compute_traffic_model_features(df_traffic_weekly_kpis,
                                                                    list_of_upgrades,
                                                                    sites_to_remove,
                                                                    remove_sites_with_more_than_one_upgrade_same_cluster,
                                                                    kpi_to_compute_upgrade_effect,
                                                                    compute_target_variable_traffic,
                                                                    cluster_key_separator=cluster_key_separator)
    # df_data_traffic_model_features.to_csv('/data/OSN/02_intermediate/df_data_traffic_model_features.csv',
    #                                          sep='|', index=False)
    # df_data_traffic_model_features = pd.read_csv('/data/ORDC/02_intermediate/df_data_traffic_model_features.csv', sep='|')

    df_capacity_kpis_features = get_capacity_kpis_features_model(list_of_upgrades,
                                                                 df_cell_affected,
                                                                 sites_to_remove,
                                                                 remove_sites_with_more_than_one_upgrade_same_cluster,
                                                                 capacity_kpis_features,
                                                                 operation_to_aggregate_cells='mean')
    # df_capacity_kpis_features.to_csv('/data/OSN/02_intermediate/df_capacity_kpis_features.csv',
    #                                         sep='|', index=False)
    # df_capacity_kpis_features = pd.read_csv('/data/ORDC/02_intermediate/df_capacity_kpis_features.csv', sep='|')

    df_upgrade_typology_features = df_upgrade_typology_features.merge(df_data_traffic_model_features,
                                                                      on = 'cluster_key',
                                                                      how = 'inner')

     ##TROUVER  df_data_traffic_model_features
    df_upgrade_typology_features['site_id'] = df_upgrade_typology_features['cluster_key'].apply(lambda x: x.split(cluster_key_separator)[0])

    # print("df_capacity_kpis_features",df_capacity_kpis_features.site_id.head())
    # print("df_upgrade_typology_features",df_upgrade_typology_features.site_id.head())
    df_upgrade_typology_features = df_upgrade_typology_features.merge(df_capacity_kpis_features,
                                                                      on = 'site_id',
                                                                      how = 'left')

    df_upgrade_typology_features = df_upgrade_typology_features.merge(list_of_upgrades[['cluster_key','tech_upgraded', 'bands_upgraded']].drop_duplicates(),
                                                                      on = 'cluster_key',
                                                                      how = 'left')

    df_upgrade_typology_features = df_upgrade_typology_features[~df_upgrade_typology_features['tech_upgraded'].isin(upgraded_to_not_consider)]

    def remove_2G_tech(tech_upgraded):
        list_tech = tech_upgraded.split("-")
        list_tech_good = []
        [list_tech_good.append(col) for col in list_tech if "2G" not in col]
        list_tech_good.sort()
        return '-'.join(list_tech_good)

    # df_upgrade_typology_features['tech_upgraded'] = df_upgrade_typology_features['tech_upgraded'].apply(remove_2G_tech)

    def remove_2G_bands(bands_upgraded):
        list_bands = bands_upgraded.split("-")
        list_bands_good = []
        [list_bands_good.append(col) for col in list_bands if "G" not in col]
        list_bands_good.sort()
        return '-'.join(list_bands_good)
    # df_upgrade_typology_features['bands_upgraded'] = df_upgrade_typology_features['bands_upgraded'].apply(remove_2G_bands)


    # if ((type_of_traffic == 'data') & group_bands):
    #
    #     ## Some manual correction to avoid only a few samples
    #     df_upgrade_typology_features['bands_upgraded'] = np.where(df_upgrade_typology_features['bands_upgraded'] == "L26-L8-U09", "L26-L8", df_upgrade_typology_features['bands_upgraded'])
    #     df_upgrade_typology_features['bands_upgraded'] = np.where(df_upgrade_typology_features['bands_upgraded'] == "L26-U09", "L26", df_upgrade_typology_features['bands_upgraded'])
    #     df_upgrade_typology_features['bands_upgraded'] = np.where(df_upgrade_typology_features['bands_upgraded'] == "L8-U09", "L8", df_upgrade_typology_features['bands_upgraded'])
    #     df_upgrade_typology_features['tech_upgraded'] = np.where(df_upgrade_typology_features['tech_upgraded'] == "3G-4G", "4G", df_upgrade_typology_features['tech_upgraded'])
    # elif ((type_of_traffic == 'voice') & group_bands):
    #     ## Some manual correction to avoid only a few samples
    #     df_upgrade_typology_features['bands_upgraded'] = np.where(df_upgrade_typology_features['bands_upgraded'] == "L26-L8-U09", "L26-U09", df_upgrade_typology_features['bands_upgraded'])
    #     df_upgrade_typology_features['bands_upgraded'] = np.where(df_upgrade_typology_features['bands_upgraded'] == "L26-L8", "L26", df_upgrade_typology_features['bands_upgraded'])
    return df_upgrade_typology_features


def train_traffic_improvement_model(df_traffic_features,
                                    type_of_traffic,
                                    remove_samples_with_target_variable_lower = True,
                                    #output_route = conf['PATH']['MODELS'],
                                    bands_to_consider = ['L18', 'L8', 'U21', 'U09'],
                                    # save_model = False,
                                    #results_route = conf['PATH']['MODELS_OUTPUT']
                                    ):
    ## Only consider those bands with enough samples to be modelled
    # df_traffic_features= df_traffic_features.loc[df_traffic_features['bands_upgraded'].isin(bands_to_consider)]
    df_traffic_features = df_traffic_features[~(df_traffic_features.bands_upgraded.isna())]
    df_traffic_features_encoded = pd.get_dummies(df_traffic_features[['tech_upgraded', 'bands_upgraded']])
    df_traffic_features = df_traffic_features.join(df_traffic_features_encoded)
    df_traffic_features.drop(columns=['tech_upgraded', 'bands_upgraded'], inplace=True)
    df_traffic_features.rename(columns={'target_variable': 'target_variable_old', 'traffic_after': 'target_variable'},
                               inplace=True)

    ## Only consider those bands with enough samples to be modelled
    #todo revoir bands to consider
    #df_traffic_features= df_traffic_features.loc[df_traffic_features['bands_upgraded'].isin(bands_to_consider)]
    #todo replace band
    print('band_to_repalce')

    #df_traffic_features['bands_upgraded'] = [str(x).replace('-L900', '') for x in df_traffic_features.bands_upgraded]
    #df_traffic_features['tech_upgraded'] = [str(x).replace('3G-4G-4G', '3G-4G') for x in df_traffic_features.tech_upgraded]
    # df_traffic_features=df_traffic_features[(~df_traffic_features.bands_upgraded.isnull()) & (~df_traffic_features.tech_upgraded.isnull())]
    # df_traffic_features_encoded = pd.get_dummies(df_traffic_features[['tech_upgraded', 'bands_upgraded']])
    # df_traffic_features = df_traffic_features.join(df_traffic_features_encoded)
    # df_traffic_features.drop(columns = ['tech_upgraded', 'bands_upgraded'],inplace = True)

    if remove_samples_with_target_variable_lower:
        if type_of_traffic == 'data':
            feature = 'total_data_traffic_dl_gb'
        else:
            feature ='total_voice_traffic_kerlands'
        df_traffic_features['is_higher'] = (df_traffic_features[feature+'_mean'] > df_traffic_features['target_variable'])
        print("Removed " + str(df_traffic_features['is_higher'].sum()/df_traffic_features['is_higher'].shape[0]) + " samples")
        df_traffic_features_higher = df_traffic_features[df_traffic_features['is_higher']]
     #df_traffic_features_higher.to_csv('/data/OSN/02_intermediate/df_traffic_features_higher'+type_of_traffic+'.csv',
     #                                        sep='|', index=False)

        #todo
        df_traffic_features = df_traffic_features[~df_traffic_features['is_higher']]
        df_traffic_features.drop(columns = ['is_higher'],inplace =True)


    cols_to_concider = [col for col in conf['TRAFFIC_IMPROVEMENT'][type_of_traffic.upper()+ '_TRAFFIC_FEATURES'] \
                        if col in df_traffic_features.columns]


    missing_cols = set(conf['TRAFFIC_IMPROVEMENT'][type_of_traffic.upper() + '_TRAFFIC_FEATURES']) - set(cols_to_concider)

    print("#########################")
    print("cols dispo :")
    print(df_traffic_features.columns)
    print("cols for Random Forest : ")
    print(cols_to_concider)

    # df_traffic_features['is_high'] = pd.qcut(df_traffic_features[feature+'_mean'], 20, labels=False)
    # cols_to_concider.append('is_high')

    X_DF = df_traffic_features[cols_to_concider]
    # if type_of_traffic == 'data':
    #     for x in ['total_data_traffic_dl_gb_mean', 'total_data_traffic_dl_gb_std', 'cell_occupation_dl_percentage3G', 'cell_occupation_dl_percentage4G']:
    #         std_value = X_DF[x].describe()[2]
    #         X_DF[x] = X_DF[x]/std_value
    # else :
    #     for x in ['total_voice_traffic_kerlands_mean', 'total_voice_traffic_kerlands_std', 'cell_occupation_dl_percentage2G', 'cell_occupation_dl_percentage3G']:
    #         std_value = X_DF[x].describe()[2]
    #         X_DF[x] = X_DF[x] / std_value
    X_DF.fillna(-1, inplace = True)
    X_DF = X_DF.append(pd.DataFrame(columns=list(missing_cols)))
    X_DF[list(missing_cols)] = 0
    X_col_name = list(X_DF.columns)

    X = np.array(X_DF)
    Y = np.array(df_traffic_features['target_variable'])

    ## get data to local training
    data=pd.concat([X_DF,df_traffic_features[['target_variable']]],axis=1)
    data.to_csv('training.csv',sep='|',index=False)
    ##------------------------------------ ##
    ## 1. Splitting between train and test  ##
    ##------------------------------------ ##

    ## Split the date between train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=50)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    ##-----------------------------##
    ## 2. Training the model       ##
    ##-----------------------------##

    logger.info('Traffic improvement  model: 2. Train the model')

    ### Training the model

        # Grid search
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'bootstrap':[True],
        'max_depth':[10,80,90,100,110,400],
        'max_features':[2,3,4,8],
        'min_samples_leaf':[2,3,4,5,10],
        'min_samples_split':[2,8,10,12,50],
        'n_estimators':[100,200,300,1000],
        'random_state':[10]
    }

    # rf = RandomForestRegressor()
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3 , n_jobs = -1 , verbose = 2)
    #
    # grid_search.fit(x_train, y_train)
    # print(grid_search.best_params_)
    #
    # def evaluate(model, test_features,test_labels):
    #     predictions = model.predict(test_features)
    #     errors = abs(predictions - test_labels)
    #     mape = 100*np.mean(errors/test_labels)
    #     accuracy = 100 - mape
    #     print("Model Performance")
    #     print('Average Error:{:0.4f} gb.'.format(np.mean(errors)))
    #     print('Accuracy = {:0.2f}%'.format(accuracy))
    #     return accuracy
    #
    # best_grid = grid_search.best_estimator_
    # grid_accuracy = evaluate(best_grid,x_test, y_test)
    # print(grid_accuracy)

    if type_of_traffic == "data":
        rf_model = RandomForestRegressor(max_depth=10, max_features=8, min_samples_leaf=2,
                      min_samples_split=2, n_estimators=300, random_state=10,bootstrap=True)
    else:
        rf_model = RandomForestRegressor(max_depth=10, max_features=8, min_samples_leaf=2,
                      min_samples_split=2, n_estimators=300, random_state=10,bootstrap=True)
    rf_model.fit(x_train, y_train)
    # print("Regenerate the best_grid")
    # print(grid_search.best_estimator_)
    # grid_search.best_estimator_.fit(x_train, y_train)

    imp = {"VarName":X_DF.columns, 'Importance':rf_model.feature_importances_}
    print(pd.DataFrame(imp))


    predictions_test =rf_model.predict(x_test)
    predictions_train =rf_model.predict(x_train)

    # MAE
    print("predictions_train: ", predictions_train)
    print("y_train: ", y_train)
    print("predictions_test: ", predictions_test)
    print("y_test: ", y_test)
    errors = abs(predictions_train - y_train)
    print('Mean Absolute error train:', round(np.mean(errors), 2))

    errors = abs(predictions_test - y_test)
    print('Mean Absolute error test:', round(np.mean(errors), 2))

    # MAPE
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')

    #R2 Score
    from sklearn.metrics import r2_score

    print("train R² : ", r2_score(y_train, predictions_train))
    print("test R² : ", r2_score(y_test,predictions_test))

    ###Crete csv file with train and test set + prediction / => models output
    df_x_test = pd.DataFrame(data=x_test, columns=X_col_name)
    df_y_test = pd.DataFrame(data= y_test, columns=['y_real'])
    df_x_train = pd.DataFrame(data=x_train, columns=X_col_name)
    df_y_train = pd.DataFrame(data=y_train, columns=['y_real'])
    df_prediction_test = pd.DataFrame(data=predictions_test, columns=['pred_y'])
    df_prediction_train = pd.DataFrame(data=predictions_train, columns=['pred_y'])
    df_output_train =  df_x_train.join(df_y_train).join(df_prediction_train)
    df_output_test =  df_x_test.join(df_y_test).join(df_prediction_test)



    # if type_of_traffic == "data" :
    #     write_csv_sql(df_output_train, "train_output_data_corrected.csv",
    #                   path="/data/OCM/05_models_output/traffic_improvment_test/", sql=False, csv=True)
    #
    #     write_csv_sql(df_output_test, "test_output_data_corrected.csv",
    #                   path="/data/OCM/05_models_output/traffic_improvment_test/", sql=False, csv=True)
    # else :
    #     write_csv_sql(df_output_train, "train_output_voice_corrected.csv",
    #                   path="/data/OCM/05_models_output/traffic_improvment_test/", sql=False, csv=True)
    #
    #     write_csv_sql(df_output_test, "test_output_voice_corrected.csv",
    #                   path="/data/OCM/05_models_output/traffic_improvment_test/", sql=False, csv=True)

    ## After checking the accuracy on the model on the test set and being sure that
    ## We are not overfitting, as we dont have many sample, the final model will be
    ## trained with the whole data

    rf_model.fit(X, Y)

    ##-----------------------------##
    ## 3. Save model               ##
    ##-----------------------------##

    # if save_model:
    #     filename = (type_of_traffic + "_traffic_improvement_random_forest" + ".sav")
    #     traffic_path_model = os.path.join(output_route,"traffic_improvement", conf['EXEC_TIME'])
    #     check_path(traffic_path_model)
    #     chdir_path(traffic_path_model)
    #     output_model = pickle.dump(rf_model, open(filename, 'wb'))
    #     logger.info('Traffic improvement: 3. Save trained model')

    ##-------------------------------------##
    ## 4. Save csv with the model results  ##
    ##-------------------------------------##
    df_output_train['origin'] = 'Train'
    df_output_test['origin'] = 'Test'
    df_output_results =  pd.concat([df_output_train,df_output_test])
    #traffic_improvement = os.path.join(results_route,'traffic_improvement',
                                        #conf['EXEC_TIME'])
    #if not os.path.exists(traffic_improvement):
    #    os.makedirs(traffic_improvement)

    #write_csv_sql(df_output_results, type_of_traffic + "_traffic_improvement_model_results.csv",
                  #path=traffic_improvement, sql=False, csv=True)

    return rf_model


def predict_traffic_improvement_model(df_traffic_features_future_upgrades,
                                      type_of_traffic = 'data',
                                      #output_route = conf['PATH']['MODELS']
                                     ):

    df_traffic_features_encoded = pd.get_dummies(df_traffic_features_future_upgrades[['tech_upgraded', 'bands_upgraded']])

    ## If the band U9 is not in the selected bands for the upgrade, we need to crease manually the column
    #if 'bands_upgraded_U09' not in df_traffic_features_encoded.columns:
    #    df_traffic_features_encoded['tech_upgraded_3G'] = 0
    #    df_traffic_features_encoded['bands_upgraded_U09'] = 0

    df_traffic_features_future_upgrades = df_traffic_features_future_upgrades.join(df_traffic_features_encoded)
    df_traffic_features_future_upgrades.drop(columns = ['tech_upgraded', 'bands_upgraded'],inplace = True)

    cols_to_concider = [col for col in conf['TRAFFIC_IMPROVEMENT'][type_of_traffic.upper() + '_TRAFFIC_FEATURES'] \
                        if col in df_traffic_features_future_upgrades.columns]

    missing_cols = set(conf['TRAFFIC_IMPROVEMENT'][type_of_traffic.upper() + '_TRAFFIC_FEATURES']) - set(cols_to_concider)

    print("cols for predict RF")
    print(cols_to_concider)
    X_DF = df_traffic_features_future_upgrades[cols_to_concider]

    # if type_of_traffic == 'data':
    #     for x in ['total_data_traffic_dl_gb_mean', 'total_data_traffic_dl_gb_std', 'cell_occupation_dl_percentage3G', 'cell_occupation_dl_percentage4G']:
    #         std_value = X_DF[x].describe()[2]
    #         X_DF[x] = X_DF[x]/std_value
    # else :
    #     for x in ['total_voice_traffic_kerlands_mean', 'total_voice_traffic_kerlands_std', 'cell_occupation_dl_percentage2G', 'cell_occupation_dl_percentage3G']:
    #         std_value = X_DF[x].describe()[2]
    #         X_DF[x] = X_DF[x] / std_value

    X_DF.fillna(-1, inplace = True)
    X_DF = X_DF.append(pd.DataFrame(columns=list(missing_cols)))
    X_DF[list(missing_cols)] = 0
    X_col_name = list(X_DF.columns)

    X = np.array(X_DF)

    last_model_path = getLastFile(os.path.join(output_route,"traffic_improvement"))

    model_path = os.path.join(output_route,"traffic_improvement", last_model_path, type_of_traffic + '_traffic_improvement_random_forest.sav')

    ##-----------------------------##
    ## 2. Reading the last version of the model ##
    ##-----------------------------##
    rf_model = pickle.load(open(model_path, 'rb'))

    predictions =rf_model.predict(X)
    df_traffic_features_future_upgrades['predictions'] = predictions
    print("colonne_to_get",df_traffic_features_future_upgrades.columns)
    if type_of_traffic =="data":
        df_traffic_features_future_upgrades['increase_of_traffic_after_the_upgrade'] = df_traffic_features_future_upgrades['predictions']-df_traffic_features_future_upgrades['total_data_traffic_dl_gb_mean']
    elif type_of_traffic == "voice":
        df_traffic_features_future_upgrades['increase_of_traffic_after_the_upgrade'] = df_traffic_features_future_upgrades['predictions']-df_traffic_features_future_upgrades['total_voice_traffic_kerlands_mean']

    df_traffic_features_future_upgrades['increase_of_traffic_after_the_upgrade'] = np.where(df_traffic_features_future_upgrades['increase_of_traffic_after_the_upgrade'] <0,0,df_traffic_features_future_upgrades['increase_of_traffic_after_the_upgrade'])
    return df_traffic_features_future_upgrades


def get_traffic_improvement(df_traffic_weekly_kpis,
                            selected_band_per_site,
                            df_traffic_features_future_upgrades,
                            kpi_to_compute_upgrade_effect=["total_data_traffic_dl_gb"]):

    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis.groupby(['date', 'week_period', 'site_id'])[
        kpi_to_compute_upgrade_effect].sum().reset_index()
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(selected_band_per_site,
                                                                    on='site_id',
                                                                    how='inner')


    kpi='total_data_traffic_dl_gb_mean'
    if( kpi not in df_traffic_features_future_upgrades.columns):
        kpi='total_voice_traffic_kerlands_mean'
    df_traffic_features_future_upgrades.to_csv('df_traffic_features_future_upgrades'+kpi+'.csv',sep='|',index=False)
    df_traffic_features_future_upgrades.drop_duplicates(subset=['site_id'],inplace=True)
    print("----df_traffic_weekly_kpis_site increase before merging-----")
    print(df_traffic_weekly_kpis_site.site_id.unique())
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.merge(
        df_traffic_features_future_upgrades[['site_id', 'increase_of_traffic_after_the_upgrade','predictions',kpi]],
        on='site_id',
        how='left')
    print("----df_traffic_weekly_kpis_site increase after traffic-----")
    print(df_traffic_features_future_upgrades.site_id.unique())

    ## Remove dismantled sites
    df_remove_sites = df_traffic_weekly_kpis.groupby(['site_id'])[kpi_to_compute_upgrade_effect].sum().reset_index()
    df_remove_sites = df_remove_sites.loc[df_remove_sites[kpi_to_compute_upgrade_effect[0]] == 0]
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[
        ~df_traffic_weekly_kpis_site['site_id'].isin(df_remove_sites['site_id'].values)]

    ## Only keep those sites with more than 10 samples

    df_remove_sites = df_traffic_weekly_kpis_site.groupby(['site_id'])[
        kpi_to_compute_upgrade_effect].count().reset_index()
    df_remove_sites = df_remove_sites.loc[df_remove_sites[kpi_to_compute_upgrade_effect[0]] < 10]
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[
        ~df_traffic_weekly_kpis_site['site_id'].isin(df_remove_sites['site_id'].values)]


    ## Remove those sites with target variable equal to na
    df_traffic_weekly_kpis_site = df_traffic_weekly_kpis_site.loc[
        ~df_traffic_weekly_kpis_site['increase_of_traffic_after_the_upgrade'].isna()]


    ## If the increase in negative, replace by 0

    df_traffic_weekly_kpis_site['increase_of_traffic_after_the_upgrade'] = np.where(df_traffic_weekly_kpis_site['increase_of_traffic_after_the_upgrade'] <0,0,df_traffic_weekly_kpis_site['increase_of_traffic_after_the_upgrade'])
    return df_traffic_weekly_kpis_site




def find_configuration_in_past_upgrades(df_data_traffic_features, df_traffic_weekly_kpis, df_sites, type = 'data'):
    """
    Function that will get the configuration on previous upgrades
    - Get rows of traffic before and after upgrade
    - Identify the upgrades (in term of configuration)
    - Split multiple upgrade in different upgrade (one per row)
    - Remove from the upgrades the one where the upgrade configuration is already existing in the before
    - Affecting the results to the output table
    """

    # explode upgraded_band for band like G1800-G900 to G1800 and G900 by line
    df_data_traffic_features=df_data_traffic_features.replace({'bands_upgraded': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                                           'F1_U2100':'U2100','F4_U2100':'U2100','F2_U900':'U900'}})
    df_data_traffic_features.drop(['tech_upgraded'],axis=1,inplace=True)
    columns= list(set(df_data_traffic_features.columns)-{'bands_upgraded'})

    band_to_tech = {"G900": '2G', 'G1800': '2G', 'U900': '3G', 'U2100': '3G', 'L800': '4G', 'L1800': '4G',
                    'L2600': '4G', 'L2300': '4G','Split1800':'4G','Split800':'4G','Split2600':'4G'}


    df_data_traffic_features = df_data_traffic_features.set_index(columns).apply(lambda x: x.str.split('-').explode()).reset_index()

    df_data_traffic_features['tech_upgraded'] = [band_to_tech[x] for x in df_data_traffic_features.bands_upgraded]
    df_data_traffic_features.drop_duplicates(inplace=True)
    df_sites=df_sites.replace({'cell_band': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                                           'F1_U2100':'U2100','F4_U2100':'U2100','F2_U900':'U900'}})


    df_before_after = pd.DataFrame(columns=['site_id', 'week_period', 'cell_name', 'cell_band'])


    for x in df_data_traffic_features.cluster_key:

        df_before = df_traffic_weekly_kpis[
            (df_traffic_weekly_kpis.site_id == x.split('-')[0]) & (
                    df_traffic_weekly_kpis.week_period == int(x.split('-')[1]) - 1)].groupby(
            ['site_id', 'week_period', 'cell_name']).cell_band.count().reset_index()
        df_before['status'] = 'before'
        df_before['key'] = x

        df_after = df_traffic_weekly_kpis[
            (df_traffic_weekly_kpis.site_id == x.split('-')[0]) & (
                    df_traffic_weekly_kpis.week_period == int(x.split('-')[1]))].groupby(
            ['site_id', 'week_period', 'cell_name']).cell_band.count().reset_index()
        df_after['status'] = 'after'
        df_after['key'] = x

        df_before_after = pd.concat([df_before_after, df_before])
        df_before_after = pd.concat([df_before_after, df_after])

    df_before_after_merge = df_before_after.merge(
        df_sites[['site_id', 'cell_name', 'cell_band']].drop_duplicates(), on=['site_id', 'cell_name'],
        how='left')
    df_before_after_merge = df_before_after_merge.merge(
        df_data_traffic_features[['cluster_key', 'tech_upgraded', 'bands_upgraded']].drop_duplicates(), left_on='key',
        right_on='cluster_key', how='left')

    df_before_after_merge.columns = ['site_id', 'week_period', 'cell_name', 'cell_band_x', 'status', 'key',
                                    'cell_band_y', 'cluster_key', 'tech_upgraded_file',
                                     'bands_upgraded_file']

    #df_before_after_merge['cell_band_and_width'] = df_before_after_merge.site_band_width + '_' + df_before_after_merge.cell_band_y
    df_before_after_merge['3G_bands'] = df_before_after_merge.cell_band_y.apply(
        lambda x: x if x in ['F1_U900','F1_U2100','F2_U900', 'F2_U2100', 'F3_U2100', 'F4_U2100','U2100','U900'] else '')
    if type == 'data':
        df_before_after_merge['4G_bands'] = df_before_after_merge.cell_band_y.apply( lambda x: x if x in ['L1800', 'L800', 'L2600'] else '')

    df_before_after_merge_groupby_cell = df_before_after_merge.groupby(['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file','cell_band_y']).cell_name.count().reset_index()

    #df_before_after_merge_groupby_cell.to_csv('/data/OSN/05_models_output/traffic_improvement/df_before_after_merge_groupby_cell.scv',sep='|')
    df_before_after_merge_groupby_cell_pivot = df_before_after_merge_groupby_cell.set_index(['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file']).pivot(
      columns='cell_band_y')['cell_name'].reset_index().rename_axis(None, axis=1)
    df_before_after_merge_groupby_cell_pivot['bands_dico'] = df_before_after_merge_groupby_cell_pivot[
         ['G1800','G900','U2100','U900']].to_dict('records')
    #['F1_U900','F1_U2100','F2_U900', 'F2_U2100', 'F3_U2100', 'F4_U2100']

    # df_before_after_merge_groupby_cell_pivot=df_before_after_merge_groupby_cell.set_index(
    #     ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file']).pivot(
    #     columns="cell_band_y")['cell_name'].reset_index().rename_axis(None, axis=1)
    print('df_before_after_merge_groupby_cell_pivot',df_before_after_merge_groupby_cell_pivot.columns)


    df_before_after_merge_groupby_3G = df_before_after_merge.groupby(['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file','3G_bands']).cell_name.count().reset_index()
    # df_before_after_merge_groupby_3G_pivot = df_before_after_merge_groupby_3G.pivot(
    #     index=['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file'],
    #     columns='3G_bands', values='cell_name').reset_index()
    df_before_after_merge_groupby_3G_pivot = df_before_after_merge_groupby_3G.set_index(['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file']).\
        pivot(columns='3G_bands')['cell_name'].reset_index().rename_axis(None, axis=1)
    df_before_after_merge_groupby_3G_pivot['3G_bands_dico'] = df_before_after_merge_groupby_3G_pivot[
         ['U2100','U900']].to_dict('records')
    #'F1_U900','F1_U2100','F2_U900', 'F2_U2100', 'F3_U2100', 'F4_U2100'
    if type == 'data':
        df_before_after_merge_groupby_4G = df_before_after_merge.groupby(
            ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
             '4G_bands']).cell_name.count().reset_index()
        df_before_after_merge_groupby_4G_pivot = df_before_after_merge_groupby_4G.set_index(
    ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file']).\
            pivot(columns='4G_bands')['cell_name'].reset_index().rename_axis(None, axis=1)
        df_before_after_merge_groupby_4G_pivot['4G_bands_dico'] = df_before_after_merge_groupby_4G_pivot[
            ['L1800', 'L800', 'L2600']].to_dict('records')

    if type =='data':
        df_before_after_merge_groupby = df_before_after_merge_groupby_cell_pivot[
            ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file','bands_dico']].merge(
            df_before_after_merge_groupby_3G_pivot[
                ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file','3G_bands_dico']],
            on=['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file'],
            how='left').reset_index().merge(df_before_after_merge_groupby_4G_pivot[
                                                ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file',
                                                 'bands_upgraded_file','4G_bands_dico']],
                                            on=['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file',
                                                'bands_upgraded_file'], how='left').reset_index()
        print(df_before_after_merge_groupby.columns)
    elif type == 'voice':
        df_before_after_merge_groupby = df_before_after_merge_groupby_cell_pivot[
            ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
             'bands_dico']].merge(
            df_before_after_merge_groupby_3G_pivot[
                ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                 '3G_bands_dico']],
            on=['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file'],
            how='left').reset_index()

    if type == 'data':
        df_before_after_merge_groupby.drop(columns=['level_0','index'], inplace=True)
    else:
        df_before_after_merge_groupby.drop(columns=['index'], inplace=True)

    df_before = df_before_after_merge_groupby[df_before_after_merge_groupby.status == 'before']
    df_after = df_before_after_merge_groupby[df_before_after_merge_groupby.status == 'after']

    if type == 'data':
        df_before.columns = ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                             'cell_band_and_width_before',
                             '3G_bands_before', '4G_bands_before']

        df_after.columns = ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                            'cell_band_and_width_after',
                            '3G_bands_after', '4G_bands_after']
    else :
        df_before.columns = ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                             'cell_band_and_width_before',
                             '3G_bands_before']

        df_after.columns = ['site_id', 'week_period', 'status', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                            'cell_band_and_width_after',
                            '3G_bands_after']
    df_before.drop(columns=['week_period', 'status'], inplace=True)
    df_after.drop(columns=['week_period', 'status'], inplace=True)

    df_before_after_merge_rename = df_before.merge(df_after,
                                                   on=['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file'],
                                                   how='left')

    list_band_upgraded = []
    list_3G_upgraded = []
    list_4G_upgraded = []

    list_3G_before = []
    list_4G_before = []

    for x in range(0, len(df_before_after_merge_rename)):
        df_test = pd.DataFrame([df_before_after_merge_rename.cell_band_and_width_before[x],
                                df_before_after_merge_rename.cell_band_and_width_after[x]])
        df_3G_upgraded = pd.DataFrame(
            [df_before_after_merge_rename['3G_bands_before'][x], df_before_after_merge_rename['3G_bands_after'][x]])
        if type =='data':
            df_4G_upgraded = pd.DataFrame(
                [df_before_after_merge_rename['4G_bands_before'][x], df_before_after_merge_rename['4G_bands_after'][x]])

        df_test_res = df_test.fillna(0).diff(axis=0).iloc[1].reset_index()
        df_3G_upgraded_res = df_3G_upgraded.fillna(0).diff(axis=0).iloc[1].reset_index()
        if type == 'data':
            df_4G_upgraded_res = df_4G_upgraded.fillna(0).diff(axis=0).iloc[1].reset_index()

        df_test_res.columns = ['band', 'band_count']
        df_3G_upgraded_res.columns = ['band', 'band_count']
        if type == 'data':
            df_4G_upgraded_res.columns = ['band', 'band_count']

        list_band_upgraded.append(list(df_test_res[df_test_res.band_count > 0].band.unique()))
        list_3G_upgraded.append(list(df_3G_upgraded_res[df_3G_upgraded_res.band_count > 0].band.unique()))

        if len(list(df_3G_upgraded_res[df_3G_upgraded_res.band_count > 0].band.unique())) > 0:
            df_3G_bef = pd.DataFrame([df_before_after_merge_rename['3G_bands_before'][x]]).fillna(0).iloc[
                0].reset_index()
            df_3G_bef.columns = ['band', 'band_count']
            list_3G_before.append(list(df_3G_bef[df_3G_bef.band_count > 0].band.unique()))
        else:
            list_3G_before.append([])

        if type == 'data':
            list_4G_upgraded.append(list(df_4G_upgraded_res[df_4G_upgraded_res.band_count > 0].band.unique()))

            if len(list(df_4G_upgraded_res[df_4G_upgraded_res.band_count > 0].band.unique())) > 0:
                df_4G_bef = pd.DataFrame([df_before_after_merge_rename['4G_bands_before'][x]]).fillna(0).iloc[
                    0].reset_index()
                df_4G_bef.columns = ['band', 'band_count']
                list_4G_before.append(list(df_4G_bef[df_4G_bef.band_count > 0].band.unique()))
            else:
                list_4G_before.append([])

    df_before_after_merge_rename['bands_upgraded_detected'] = list_band_upgraded
    df_before_after_merge_rename['3G_before'] = list_3G_before
    df_before_after_merge_rename['3G_upgrades'] = list_3G_upgraded
    if type == 'data':
        df_before_after_merge_rename['4G_before'] = list_4G_before
        df_before_after_merge_rename['4G_upgrades'] = list_4G_upgraded

    # Code to have one row per upgrade
    df_output = pd.DataFrame()
    for x in range(0, len(df_before_after_merge_rename)):
        if type == 'data':
            if len(df_before_after_merge_rename['3G_upgrades'][x]) >= 1:
                if len(df_before_after_merge_rename['3G_upgrades'][x]) == 1:
                    df_output_3G = df_before_after_merge_rename[
                                       ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                        'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                        'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                        'bands_upgraded_detected', '3G_before']].iloc[x:x + 1, :]
                    df_output_3G['3G_upgrades'] = df_before_after_merge_rename['3G_upgrades'][x][0]
                    df_output_3G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                            'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                    df_output_3G['tech_upgrades'] = '3G'
                    df_output = pd.concat([df_output, df_output_3G])
                elif len(df_before_after_merge_rename['3G_upgrades'][x]) > 1:
                    for i in range(0, len(df_before_after_merge_rename['3G_upgrades'][x])):
                        df_output_3G = df_before_after_merge_rename[
                                           ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                            'bands_upgraded_detected', '3G_before']].iloc[x:x + 1, :]
                        df_output_3G['3G_upgrades'] = df_before_after_merge_rename['3G_upgrades'][x][i]
                        df_output_3G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                                'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                                'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                                'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                        df_output_3G['tech_upgrades'] = '3G'
                        df_output = pd.concat([df_output, df_output_3G])
        else:
            if len(df_before_after_merge_rename['3G_upgrades'][x]) >= 1:
                if len(df_before_after_merge_rename['3G_upgrades'][x]) == 1:
                    df_output_3G = df_before_after_merge_rename[
                                       ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                        'cell_band_and_width_before', '3G_bands_before',
                                        'cell_band_and_width_after', '3G_bands_after',
                                        'bands_upgraded_detected', '3G_before']].iloc[x:x + 1, :]
                    df_output_3G['3G_upgrades'] = df_before_after_merge_rename['3G_upgrades'][x][0]
                    df_output_3G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after',
                                            'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                    df_output_3G['tech_upgrades'] = '3G'
                    df_output = pd.concat([df_output, df_output_3G])
                elif len(df_before_after_merge_rename['3G_upgrades'][x]) > 1:
                    for i in range(0, len(df_before_after_merge_rename['3G_upgrades'][x])):
                        df_output_3G = df_before_after_merge_rename[
                                           ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after',
                                            'bands_upgraded_detected', '3G_before']].iloc[x:x + 1, :]
                        df_output_3G['3G_upgrades'] = df_before_after_merge_rename['3G_upgrades'][x][i]
                        df_output_3G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                                'cell_band_and_width_before', '3G_bands_before',
                                                'cell_band_and_width_after', '3G_bands_after',
                                                'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                        df_output_3G['tech_upgrades'] = '3G'
                        df_output = pd.concat([df_output, df_output_3G])
        if type == 'data':
            if len(df_before_after_merge_rename['4G_upgrades'][x]) >= 1:
                if len(df_before_after_merge_rename['4G_upgrades'][x]) == 1:
                    df_output_4G = df_before_after_merge_rename[
                                       ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                        'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                        'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                        'bands_upgraded_detected', '4G_before']].iloc[x:x + 1, :]
                    df_output_4G['4G_upgrades'] = df_before_after_merge_rename['4G_upgrades'][x][0]
                    df_output_4G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                            'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                    df_output_4G['tech_upgrades'] = '4G'
                    df_output = pd.concat([df_output, df_output_4G])
                elif len(df_before_after_merge_rename['4G_upgrades'][x]) > 1:
                    for i in range(0, len(df_before_after_merge_rename['4G_upgrades'][x])):
                        df_output_4G = df_before_after_merge_rename[
                                           ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                            'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                            'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                            'bands_upgraded_detected', '4G_before']].iloc[x:x + 1, :]
                        df_output_4G['4G_upgrades'] = df_before_after_merge_rename['4G_upgrades'][x][i]
                        df_output_4G.columns = ['site_id', 'key', 'tech_upgraded_file', 'bands_upgraded_file',
                                                'cell_band_and_width_before', '3G_bands_before', '4G_bands_before',
                                                'cell_band_and_width_after', '3G_bands_after', '4G_bands_after',
                                                'bands_upgraded_detected', 'bands_before', 'bands_upgrades']
                        df_output_4G['tech_upgrades'] = '4G'
                        df_output = pd.concat([df_output, df_output_4G])

    # Remove from this table the upgrades where the "bands_upgrades" is already existing in "bands_before"
    df_output['keep_upgrades'] = df_output[['bands_upgrades', 'bands_before']].apply(
        lambda x: 'no' if x.iloc[0] in x.iloc[1] else 'yes', axis=1)
    df_output_upgrade_to_keep = df_output[df_output.keep_upgrades == 'yes']

    # Preparing the merge
    df_output_upgrade_to_keep_selected_columns = df_output_upgrade_to_keep[['key', 'bands_upgrades', 'tech_upgrades']]
    df_output_upgrade_to_keep_selected_columns.columns = ['cluster_key', 'bands_upgraded', 'tech_upgraded']

    df_data_traffic_features.rename(
        columns={'bands_upgraded': 'bands_upgraded_old', 'tech_upgraded': 'tech_upgraded_old'}, inplace=True)

    df_data_traffic_features_merge = df_data_traffic_features.merge(df_output_upgrade_to_keep_selected_columns,
                                                                    how='left', on='cluster_key')

    return df_before_after_merge, df_data_traffic_features_merge

def cells_per_upgrade(df_before_after_merge, df_data_traffic_features_merge, type = 'data'):
    """
    Get the list of cell_id to consider (per upgrade) to caculate the traffic_after
    - Get subset of data for the upgrade
    - Catch cells_id for "before"

    """
    ### Obtention de la liste de cell_id a prendre en compte (propre a chaque upgrade) pour le traffic after
    list_cell_after = []

    for x in range(0, len(df_data_traffic_features_merge)):
        cluster_key = df_data_traffic_features_merge.loc[x]['cluster_key']
        df_per_action = df_before_after_merge[df_before_after_merge.cluster_key.isin([cluster_key])]
        df_per_action_list_cell_before = list(df_per_action[df_per_action.status == 'before'].cell_name)
        if df_data_traffic_features_merge.loc[x]['tech_upgraded'] == '3G':
            tech_upgraded = df_data_traffic_features_merge.loc[x]['bands_upgraded']
            df_per_action_list_cell_after = list(df_per_action[(df_per_action.status == 'after') & (
                df_per_action['3G_bands'].isin([tech_upgraded]))].cell_name)
        elif (df_data_traffic_features_merge.loc[x]['tech_upgraded'] == '4G') & (type == 'data'):
            tech_upgraded = df_data_traffic_features_merge.loc[x]['bands_upgraded']
            df_per_action_list_cell_after = list(df_per_action[(df_per_action.status == 'after') & (
                df_per_action['4G_bands'].isin([tech_upgraded]))].cell_name)
        else:
            df_per_action_list_cell_after = []

        df_per_action_list_cell_after = df_per_action_list_cell_before + df_per_action_list_cell_after

        list_cell_after.append(df_per_action_list_cell_after)

    df_data_traffic_features_merge['cell_after'] = list_cell_after

    return df_data_traffic_features_merge


def traffic_after_per_action(df_data_traffic_features_merge,
                             df_traffic_weekly_kpis,
                             kpi_to_compute_upgrade_effect = ['total_data_traffic_dl_gb']):
    """
    Function that will calculate the traffic after per action of upgrade
    """
    ### Calcul du traffic after par action
    df_before_actions = []
    df_after_actions = []
    for x in range(0, len(df_data_traffic_features_merge)):
        site_id = df_data_traffic_features_merge.loc[x]['site_id']
        cluster_key = df_data_traffic_features_merge.iloc[x]['cluster_key']
        week_of_upgrade = df_data_traffic_features_merge.iloc[x]['cluster_key'].split('-')[1]
        cells_after = df_data_traffic_features_merge.iloc[x]['cell_after']

        df_traffic_weekly_kpis_site = \
        df_traffic_weekly_kpis[df_traffic_weekly_kpis.site_id.isin([site_id])].groupby(['site_id',
                                                                                        'cell_name',
                                                                                        'date',
                                                                                        'week_period'])[
            kpi_to_compute_upgrade_effect].sum().reset_index()

        df_traffic_weekly_kpis_site['cluster_key'] = cluster_key
        df_traffic_weekly_kpis_site['week_of_the_upgrade'] = week_of_upgrade

        df_traffic_weekly_kpis_cluster_tech = df_traffic_weekly_kpis_site.groupby(['cluster_key',
                                                                                   'cell_name',
                                                                                   'date',
                                                                                   'week_period',
                                                                                   'week_of_the_upgrade'])[
            kpi_to_compute_upgrade_effect].sum().reset_index()

        df_traffic_weekly_kpis_cluster_tech['week_of_the_upgrade'] = df_traffic_weekly_kpis_cluster_tech[
            'week_of_the_upgrade'].apply(int).apply(str)
        df_traffic_weekly_kpis_cluster_tech['week_period'] = df_traffic_weekly_kpis_cluster_tech['week_period'].apply(
            int).apply(str)

        df_traffic_weekly_kpis_cluster_tech['lag_between_upgrade'] = df_traffic_weekly_kpis_cluster_tech[
            ['week_of_the_upgrade', 'week_period']].apply(
            lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

        df_before = df_traffic_weekly_kpis_cluster_tech[
            (df_traffic_weekly_kpis_cluster_tech['lag_between_upgrade'] >= -8) & (
                    df_traffic_weekly_kpis_cluster_tech['lag_between_upgrade'] < 0)]
        df_before_group = df_before.groupby(['cluster_key',
                                             'date',
                                             'week_period',
                                             'week_of_the_upgrade'])[kpi_to_compute_upgrade_effect].sum().reset_index()
        df_before_actions.append(df_before_group[kpi_to_compute_upgrade_effect].mean()[0])

        ## Compute target variable: average of the traffic after the upgrade
        df_target = df_traffic_weekly_kpis_cluster_tech[
            (df_traffic_weekly_kpis_cluster_tech['lag_between_upgrade'] > conf['TRAFFIC_IMPROVEMENT'][
                'WEEKS_TO_WAIT_AFTER_UPGRADE']) & (
                    df_traffic_weekly_kpis_cluster_tech['lag_between_upgrade'] <= 8)]
        ## Filter on the cells that act on the upgrade
        df_target_selected_cells = df_target[df_target.cell_name.isin(cells_after)]
        df_target_group = df_target_selected_cells.groupby(['cluster_key',
                                                            'date',
                                                            'week_period',
                                                            'week_of_the_upgrade'])[
            kpi_to_compute_upgrade_effect].sum().reset_index()
        df_after_actions.append(df_target_group[kpi_to_compute_upgrade_effect].mean()[0])

    df_data_traffic_features_merge['traffic_before'] = df_before_actions
    df_data_traffic_features_merge['traffic_after'] = df_after_actions
    print(df_data_traffic_features_merge.head())

    # df_number_upgrade = df_data_traffic_features_merge.groupby('cluster_key').apply(
    #     lambda x: len(x.bands_upgraded.unique())).reset_index()
    #
    # df_number_upgrade.columns = ['cluster_key', 'number_band']
    # df_data_traffic_features_merge = df_data_traffic_features_merge.merge(df_number_upgrade, how='left', on='cluster_key')

   # if('total_data_traffic_dl_gb_mean' in df_data_traffic_features_merge.columns):
    #    df_data_traffic_features_merge['traffic_before'] = df_data_traffic_features_merge['total_data_traffic_dl_gb_mean'] / \
    #                                          df_data_traffic_features_merge['number_band']
    return df_data_traffic_features_merge

def compute_traffic_after(df_data_traffic_features, df_traffic_weekly_kpis,kpi_to_compute_upgrade_effect='total_data_traffic_dl_gb'):
    df_data_traffic_features = df_data_traffic_features.replace(
        {'bands_upgraded': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                            'F1_U2100': 'U2100', 'F4_U2100': 'U2100', 'F2_U900': 'U900'}})
    df_traffic_weekly_kpis = df_traffic_weekly_kpis.replace(
        {'cell_band': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                            'F1_U2100': 'U2100', 'F4_U2100': 'U2100', 'F2_U900': 'U900'}})
    df_traffic_weekly_kpis.week_period=df_traffic_weekly_kpis.week_period.astype(str)
    df_data_traffic_features['week_period'] = [x.split('-')[1] for x in df_data_traffic_features.cluster_key]

    df_data_traffic_features.drop(['tech_upgraded'], axis=1, inplace=True)
    columns = list(set(df_data_traffic_features.columns) - {'bands_upgraded'})

    band_to_tech = {"G900": '2G', 'G1800': '2G', 'U900': '3G', 'U2100': '3G', 'L800': '4G', 'L1800': '4G',
                    'L2600': '4G', 'L2300': '4G', 'Split1800': '4G', 'Split800': '4G', 'Split2600': '4G'}

    df_data_traffic_features = df_data_traffic_features.set_index(columns).apply(
        lambda x: x.str.split('-').explode()).reset_index()

    df_data_traffic_features['tech_upgraded'] = [band_to_tech[x] for x in df_data_traffic_features.bands_upgraded]
    df_data_traffic_features.drop_duplicates(inplace=True)



    for index, row in df_data_traffic_features.iterrows():
        # df_interm=oss_counter[(oss_counter.site_id==row['site_id'])& (oss_counter.week_period==row['week_period']) &(oss_counter.cell_band.isin(list(set(row['bands_upgraded_list'])-set(row['bands_upgraded'])))) ]
        df_for_site = df_traffic_weekly_kpis[(df_traffic_weekly_kpis.site_id == row['site_id'])]

        cell_before = list(df_for_site[df_for_site.week_period < row['week_period']].cell_name.unique())

        all_before_with_bands_upgraded = df_for_site[df_for_site.week_period == row['week_period']].cell_name.unique()

        df_upgraded_bands_cells = set(all_before_with_bands_upgraded) - set(cell_before)

        current_band_ugraded_cells = list(df_for_site[(df_for_site.cell_band == row[
            'bands_upgraded']) & df_for_site.cell_name.isin(df_upgraded_bands_cells) & (df_for_site.week_period == row[
            'week_period'])].cell_name.unique())

        cell_to_compute_after = cell_before + current_band_ugraded_cells

        df_oss_after=df_for_site[df_for_site.cell_name.isin(cell_to_compute_after)]
        df_target = df_oss_after.groupby('week_period')[kpi_to_compute_upgrade_effect].sum().reset_index()
        df_target['lag_between_upgrade'] = [get_lag_between_two_week_periods(row['week_period'], x) for x in
                                            df_target.week_period]

        traffic_affer = df_target[(df_target.lag_between_upgrade > 4) & (df_target.lag_between_upgrade <= 8)][
            kpi_to_compute_upgrade_effect].mean()

        df_data_traffic_features.loc[index, 'traffic_after'] = traffic_affer

    return  df_data_traffic_features




if __name__=='__main__':
    df_data_traffic_features=pd.read_csv('/data/OSN/05_models_output/traffic_improvement/file_for_test/df_data_traffic_features.csv',sep='|')
    df_voice_traffic_features = pd.read_csv(
        '/data/OSN/05_models_output/traffic_improvement/file_for_test/df_voice_traffic_features.csv', sep='|')
    df_traffic_weekly_kpis=pd.read_csv('/data/OSN/02_intermediate/Thies/preprocessed_oss_counter_all_v3.csv',sep='|')
    df_sites=pd.read_csv('/data/OSN/02_intermediate/Thies/df_sites.csv',sep='|')
    df1,df2=find_configuration_in_past_upgrades(df_data_traffic_features, df_traffic_weekly_kpis, df_sites, type='data')
    df3, df4 = find_configuration_in_past_upgrades(df_voice_traffic_features, df_traffic_weekly_kpis, df_sites,
                                                   type='voice')
