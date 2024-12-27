import os
import pandas as pd
from src.d00_conf.conf import conf
from src.d01_utils.utils import write_csv_sql, get_month_year_period, add_logging_info


@add_logging_info
def compute_site_margin(revenues_per_unit_traffic):
    """
    The compute_site_margin function calculates the yearly site margin by subtracting operational
    expenses from total site revenues, grouping the results by cluster,
    and saving the output as a CSV file.

    Parameters
    ----------
    revenues_per_unit_traffic : pd.DataFrame
        A pandas DataFrame containing columns cluster_key, revenue_total_site, and cout_interco.
    Returns
    -------
    df_margin_per_site: pd.DataFrame
        A pandas DataFrame with columns cluster_key and site_margin_yearly.
    """
    revenues_per_unit_traffic.rename(columns={'cout_interco': 'opex_clients'}, inplace=True)
    revenues_per_unit_traffic['site_margin_monthly'] = (
            revenues_per_unit_traffic['revenue_total_site'] -
            revenues_per_unit_traffic['opex_clients'])

    df_margin_per_site = revenues_per_unit_traffic.groupby(['cluster_key'])[
        ['site_margin_monthly']].sum().reset_index()
    df_margin_per_site.columns = ['cluster_key', 'site_margin_yearly']

    margin_path = os.path.join(conf['PATH']['MODELS_OUTPUT'],
                               "site_margin",
                               conf['EXEC_TIME'])

    if not os.path.exists(margin_path):
        os.makedirs(margin_path)
    write_csv_sql(df=df_margin_per_site,
                  name='margin_per_site_and_service_' + conf["USE_CASE"] + '_from_capacity.csv',
                  path=margin_path,
                  sql=False,
                  csv=True)

    return df_margin_per_site


@add_logging_info
def compute_increase_of_yearly_site_margin(revenues_per_unit_traffic,
                                           df_increase_arpu_due_to_the_upgrade,
                                           df_margin_per_site):
    """
    The compute_increase_of_yearly_site_margin function calculates the increase in yearly site
    margin due to an upgrade. It processes data on revenues per unit traffic,
    ARPU increases due to the upgrade, and site margins to compute the increase in yearly revenues
    and margins. The results are saved as a CSV file.

    Parameters
    ----------
    revenues_per_unit_traffic: pd.DataFrame
        The revenues per unit
    df_increase_arpu_due_to_the_upgrade: pd.DataFrame
        DataFrame containing ARPU increases due to the upgrade.
    df_margin_per_site: pd.DataFrame
        DataFrame containing margin data per site.
    Returns
    -------
    df_increase_arpu_by_year: pd.DataFrame
        DataFrame containing the computed increase in yearly site margin
    """
    # Compute the yearly increase in arpu for both the data and voice services
    #df_increase_arpu_by_year = df_increase_arpu_due_to_the_upgrade.groupby(
    #    ['site_id', 'year'])[[
    #    'arpu_increase_due_to_the_upgrade_data_xof']].sum().reset_index()

    df_increase_arpu_by_year = df_increase_arpu_due_to_the_upgrade.groupby(
        ['cluster_key', 'year'])[[
        'arpu_increase_due_to_the_upgrade_data', 'arpu_increase_due_to_the_upgrade_voice']
    ].sum().reset_index()

    # Number of months per year to compute the opex costs
    df_increase_arpu_due_to_the_upgrade['date'] = pd.to_datetime(
        df_increase_arpu_due_to_the_upgrade['date'])
    df_increase_arpu_due_to_the_upgrade['month_period'] = (
        df_increase_arpu_due_to_the_upgrade['date'].apply(get_month_year_period))
    df_months_per_year = (
        df_increase_arpu_due_to_the_upgrade[['cluster_key', 'year',
                                             'month_period']].drop_duplicates().groupby(
            ['cluster_key', 'year'])['month_period'].count().reset_index())
    df_months_per_year.columns = ['cluster_key', 'year', 'number_of_months_per_year']
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_months_per_year,
                                                              on=['cluster_key', 'year'],
                                                              how='left')

    # Compute the annual revenues by service
    # We already have 24 months of data. So the annual margin is done by summing the margin of the
    # 12 most recent ones
    revenues_per_unit_traffic['revenue_data'] = (revenues_per_unit_traffic.revenue_data_mobile +
                                                 revenues_per_unit_traffic.revenue_data_box)
    revenues_per_unit_traffic['revenue_voice'] = revenues_per_unit_traffic.revenue_voice_mobile
   #df_annual_revenues_by_service = revenues_per_unit_traffic.groupby(
   #    ['site_id'])[['revenue_data']].sum().reset_index()
    df_annual_revenues_by_service = revenues_per_unit_traffic.groupby(
        ['cluster_key'])[['revenue_data', 'revenue_voice']].sum().reset_index()

    df_annual_revenues_by_service.columns = ['cluster_key', 'annual_revenues_data_traffic',
                                             'annual_revenues_voice_traffic']



    # Merge with the increase in arpu due to the upgrade
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_annual_revenues_by_service,
                                                              on='cluster_key',
                                                              how='left')

    df_increase_arpu_by_year['increase_yearly_data_revenues'] = (
            df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_data'] /
            df_increase_arpu_by_year['annual_revenues_data_traffic'])

    df_increase_arpu_by_year['increase_yearly_voice_revenues'] = (
            df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_voice'] /
            df_increase_arpu_by_year['annual_revenues_voice_traffic'])

    df_increase_arpu_by_year['increase_yearly_revenues'] = \
        (df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_data'] +
         df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_voice']) / (
                df_increase_arpu_by_year['annual_revenues_data_traffic'] +
                df_increase_arpu_by_year['annual_revenues_voice_traffic'])

    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(
        df_margin_per_site[['cluster_key', 'site_margin_yearly']],
        on='cluster_key',
        how='left')

    # Compute the increase in yearly data and voice margin due to the upgrade
    df_increase_arpu_by_year['increase_yearly_margin_due_to_the_upgrade'] = \
        df_increase_arpu_by_year[
            'increase_yearly_revenues'] * \
        df_increase_arpu_by_year[
            'site_margin_yearly']

    increase_margin_path = os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                        "increase_site_margin",
                                        conf['EXEC_TIME'])

    if not os.path.exists(increase_margin_path):
        os.makedirs(increase_margin_path)
    write_csv_sql(df=df_increase_arpu_by_year,
                  name='increase_margin_per_site_and_service_' + conf[
                      "USE_CASE"] + '_from_capacity.csv',
                  path=increase_margin_path,
                  sql=False, csv=True)

    return df_increase_arpu_by_year
