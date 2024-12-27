
from src.conf import conf, conf_loader

conf_loader("OSN")
# from src.d01_utils.utils import *


def get_affected_cells_with_interactions_between_upgrades(df_traffic_weekly_kpis):
    df_cell_affected, df_cell_both_affected = get_affected_cells(df_traffic_weekly_kpis)
    while df_cell_both_affected.shape[0] > 0:
        df_cell_both_affected.drop(columns=['starting_week_site', 'starting_week_cell', 'is_upgrade'], inplace=True)
        df_cell_affected_2, df_cell_both_affected = get_affected_cells(df_cell_both_affected)
        df_cell_affected = pd.concat([df_cell_affected, df_cell_affected_2])

    # Merge upgrades with less than 4 weeks of difference
    def correct_sites_with_several_deployment(df):
        df['starting_week_site'] = df['starting_week_site'].min()
        number_of_upgrades = df['week_of_the_upgrade'].drop_duplicates().values
        number_of_upgrades.sort()
        for i in range(0, len(number_of_upgrades) - 1):
            if abs(get_lag_between_two_week_periods(str(number_of_upgrades[i]), str(number_of_upgrades[i + 1]))) < \
                    conf['TRAFFIC_IMPROVEMENT']['MAXIMUM_WEEKS_TO_GROUP_UPGRADES']:
                # Remove iteration between different upgrades if there is less than 4 weeks
                # bands_to_eliminate = df[df['week_of_the_upgrade'] == number_of_upgrades[i]][
                #     'bands_upgraded'].drop_duplicates()
                # bands_to_eliminate = bands_to_eliminate.values[0].split("-")
                # df = df[~((df['cell_band'].isin(bands_to_eliminate)) & (
                #             df['week_of_the_upgrade'] == number_of_upgrades[i + 1]))]
                df['week_of_the_upgrade'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i],
                                                     number_of_upgrades[i + 1], df['week_of_the_upgrade'])

                new_bands = "-".join(df['bands_upgraded'].drop_duplicates())
                new_bands = '-'.join(sorted(np.unique(new_bands.split('-'))))
                new_tech = "-".join(df['tech_upgraded'].drop_duplicates())
                new_tech = '-'.join(sorted(np.unique(new_tech.split('-'))))
                df['tech_upgraded'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i + 1], new_tech,
                                               df['tech_upgraded'])
                df['bands_upgraded'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i + 1], new_bands,
                                                df['bands_upgraded'])
                df = df.drop_duplicates()
        return df

    df_cell_affected = df_cell_affected.groupby('site_id').apply(correct_sites_with_several_deployment).reset_index(
        drop=True)
    return df_cell_affected


# def get_affected_cells_with_interactions_between_upgrades(df_traffic_weekly_kpis):
#     df_cell_affected = get_affected_cells(df_traffic_weekly_kpis)
#     ## Merge upgrades with less than 4 weeks of difference
#     def correct_sites_with_several_deployment(df):
#         number_of_upgrades = df['week_of_the_upgrade'].drop_duplicates().values
#         number_of_upgrades.sort()
#         for i in range(0,len(number_of_upgrades)-1):
#             if abs(get_lag_between_two_week_periods(str(number_of_upgrades[i]), str(number_of_upgrades[i+1]))) < conf['TRAFFIC_IMPROVEMENT']['MAXIMUM_WEEKS_TO_GROUP_UPGRADES']:
#                 ## Remove iteration between different upgrades if there is less than 4 weeks
#                 bands_to_eliminate = df[df['week_of_the_upgrade'] ==number_of_upgrades[i]]['bands_upgraded'].drop_duplicates()
#                 bands_to_eliminate = bands_to_eliminate.values[0].split("-")
#                 df = df[~((df['cell_band'].isin(bands_to_eliminate)) & (df['week_of_the_upgrade']==number_of_upgrades[i+1]))]
#                 df['week_of_the_upgrade'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i], number_of_upgrades[i+1], df['week_of_the_upgrade'])
#
#                 new_bands = "-".join(df['bands_upgraded'].drop_duplicates())
#                 new_tech = "-".join(df['tech_upgraded'].drop_duplicates())
#                 df['tech_upgraded'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i+1], new_tech, df['tech_upgraded'])
#                 df['bands_upgraded'] = np.where(df['week_of_the_upgrade'] == number_of_upgrades[i+1], new_bands, df['bands_upgraded'])
#                 df = df.drop_duplicates()
#         return df
#     df_cell_affected = df_cell_affected.groupby('site_id').apply(correct_sites_with_several_deployment).reset_index(drop = True)
#     return df_cell_affected


def get_affected_cells(df_traffic_weekly_kpis):
    """
    Function with the cells that were affected by an upgrade in the Past year:

    :return list of cells that were affected:

    """
    df_traffic_weekly_kpis=df_traffic_weekly_kpis[~((df_traffic_weekly_kpis['cell_band'].isnull()) | (df_traffic_weekly_kpis['cell_band'] == 'REPLACE'))]
    df_traffic_weekly_kpis.reset_index(drop=True,inplace=True)
    df_traffic_weekly_kpis = df_traffic_weekly_kpis[~(df_traffic_weekly_kpis['cell_band'] == "L23")]

    ## Remove sites that have NA
    # df_bad_sites = df_traffic_weekly_kpis[df_traffic_weekly_kpis['cell_occupation_dl_percentage'].isna()][
    #     'site_id'].drop_duplicates()

    # df_traffic_weekly_kpis = df_traffic_weekly_kpis.loc[~df_traffic_weekly_kpis['site_id'].isin(df_bad_sites.values)]

    ### Compute the starting date of a site
    df_starting_date_site = df_traffic_weekly_kpis.groupby('site_id')['week_period'].min().reset_index()
    df_starting_date_site.columns = ['site_id', 'starting_week_site']

    df_traffic_weekly_kpis = df_traffic_weekly_kpis.merge(df_starting_date_site,
                                                          on='site_id',
                                                          how='left')

    ## Compute the starting date of a cell
    df_affected = df_traffic_weekly_kpis.groupby('cell_name')['week_period'].min().reset_index()

    df_affected.columns = ['cell_name',
                           'starting_week_cell']

    df_traffic_weekly_kpis = df_traffic_weekly_kpis.merge(df_affected,
                                                          on='cell_name',
                                                          how='left')

    ## Get cells affected by an upgrade
    def is_upgrade(starting_week_cell, starting_week_site):
        if starting_week_site < starting_week_cell:
            return 1
        return 0

    df_traffic_weekly_kpis['is_upgrade'] = df_traffic_weekly_kpis[['starting_week_cell', 'starting_week_site']].apply(lambda x: is_upgrade(x.iloc[0], x.iloc[1]), axis=1)

    print(df_traffic_weekly_kpis['is_upgrade'].sum())

    ## Filter upgraded and not upgraded cells
    df_cell_upgraded = df_traffic_weekly_kpis.loc[df_traffic_weekly_kpis['is_upgrade'] == 1]

    print(df_cell_upgraded['is_upgrade'].sum())
    print(df_cell_upgraded['site_id'].nunique())
    print(df_cell_upgraded['cell_name'].nunique())

    df_cell_not_upgraded = df_traffic_weekly_kpis.loc[df_traffic_weekly_kpis['is_upgrade'] == 0]

    ## Get all the technologies and bands in the upgrade
    df_site_upgraded = df_cell_upgraded[['site_id', 'starting_week_cell', 'cell_band']].drop_duplicates()

    df_site_upgraded = df_site_upgraded.groupby(['site_id', 'starting_week_cell'])['cell_band'].apply(
        lambda x: '-'.join(x)).reset_index()
    df_site_upgraded['is_affected'] = 1
    df_site_upgraded.columns = ['site_id', 'week_of_the_upgrade', 'bands_upgraded', 'is_affected']
    df_site_upgraded_tech = df_cell_upgraded[['site_id', 'starting_week_cell', 'cell_tech']].drop_duplicates()
    df_site_upgraded_tech = df_site_upgraded_tech.groupby(['site_id', 'starting_week_cell'])['cell_tech'].apply(
        lambda x: '-'.join(x)).reset_index()
    df_site_upgraded_tech.columns = ['site_id', 'week_of_the_upgrade', 'tech_upgraded']
    df_site_upgraded = df_site_upgraded.merge(df_site_upgraded_tech,
                                              on=['site_id', 'week_of_the_upgrade'],
                                              how='left')

    ## Cells that are an upgrade can be affected by another upgrade -> when has been more than 1 upgrade on the site on the yeawr
    number_of_upgrades_by_site = \
        df_site_upgraded[['site_id', 'bands_upgraded']].drop_duplicates().groupby(['site_id'])[
            'bands_upgraded'].count().reset_index()
    print(number_of_upgrades_by_site.head())

    number_of_upgrades_by_site = number_of_upgrades_by_site[number_of_upgrades_by_site['bands_upgraded'] > 1]
    ## Get the interaction between cells that are both upgraded and an upgrade
    df_cell_both_upgraded_and_upgrades = df_cell_upgraded[
        df_cell_upgraded['site_id'].isin(number_of_upgrades_by_site['site_id'])]
    print(df_cell_both_upgraded_and_upgrades.head())


    # Cell affected by an upgrade -> cells not upgrades and sites in the list of sites upgraded
    df_cell_affected_by_an_upgrade = df_cell_not_upgraded[df_cell_not_upgraded['site_id'].isin(df_site_upgraded['site_id'])]

    ## It will duplicate the cell info for each upgrade
    df_cell_affected_by_an_upgrade = df_cell_affected_by_an_upgrade.merge(df_site_upgraded,
                                                                          left_on=['site_id'],
                                                                          right_on=['site_id'],
                                                                          how='right')

    # df_cell_affected_by_an_upgrade.to_csv('/data/OSN/02_intermediate/file_for_test/df_cell_affected_by_an_upgrade.csv', index = False, sep = '|')
    # df_cell_both_upgraded_and_upgrades.to_csv('/data/OSN/02_intermediate/file_for_test/df_cell_both_upgraded_and_upgrades.csv', index = False, sep = '|')

    return df_cell_affected_by_an_upgrade, df_cell_both_upgraded_and_upgrades
#
# def get_affected_cells(df_traffic_weekly_kpis, path=conf['PATH']['INTERMEDIATE_DATA']):
#     """
#         This function process past upgrades performed by ORDC.
#         As we didn't find any upgrade automatically, a list of pasts upgrades have been sent to tag theses sites.
#
#         - Load data frame of past upgrades
#         - Compute week of the upgrade from date of effective action
#         _ Add oss kpi to data
#         """
#     # Load data frame of past upgrades
#     df_past_upgrades = pd.read_csv(os.path.join(path, 'past_upgrades.csv'), sep=';',
#                                    parse_dates=['date_prevue', 'date_effective'], encoding='ISO-8859-1')
#
#     # Compute week of the upgrade from date of effective action
#     df_past_upgrades['week_of_the_upgrade'] = df_past_upgrades.date_effective.apply(
#         lambda x: str(x.isocalendar()[0]) + str(x.isocalendar()[1]))
#
#     # Add oss kpi to data
#     df_past_upgrades.rename(columns={"site_id": 'ids', "site_code": 'site_id'}, inplace=True)
#
#     df_past_upgrades = df_past_upgrades.merge(df_traffic_weekly_kpis, on=['site_id'], how='inner')
#
#     return df_past_upgrades
