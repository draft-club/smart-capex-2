from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_increase_of_yearly_site_margin(
    project_id: str,
    location: str,
    increase_arpu_by_year_table_id: str,
    revenues_per_unit_traffic_data_input: Input[Dataset],
    increase_arpu_due_to_the_upgrade_data_input: Input[Dataset],
    margin_per_site_data_input: Input[Dataset],
    increase_arpu_by_year_data_output: Output[Dataset]):
    """It computes the increase in yearly margin per site due to ARPU (Average Revenue Per User) upgrades.

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        increase_arpu_by_year_table_id (str): It holds the resource name on BigQuery
        revenues_per_unit_traffic_data_input (Input[Dataset]): It holds revenues per unit traffic data
        increase_arpu_due_to_the_upgrade_data_input (Input[Dataset]): It holds increase arpu due to the upgrade data
        margin_per_site_data_input (Input[Dataset]): It holds the computed yearly margin per site data
        increase_arpu_by_year_data_output (Output[Dataset]): It holds computed yearly increase in arpu and margins per site

    Returns:
        increase_arpu_by_year_data_output (Output[Dataset]): It holds computed yearly increase in arpu and margins per site
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    def get_month_year_period(d):
        if pd.isnull(d):
            return d
        month = f"{d.month:02d}"
        year = f"{d.year:04d}"
        return year + month


    # Load Data
    revenues_per_unit_traffic = pd.read_parquet(revenues_per_unit_traffic_data_input.path)
    df_increase_arpu_due_to_the_upgrade = pd.read_parquet(increase_arpu_due_to_the_upgrade_data_input.path)
    df_margin_per_site = pd.read_parquet(margin_per_site_data_input.path)

    # Compute the yearly increase in arpu for both the data and voice services
    df_increase_arpu_by_year = (df_increase_arpu_due_to_the_upgrade
                                .groupby(['site_id', 'bands_upgraded', 'site_area', 'year'])
                                [['arpu_increase_due_to_the_upgrade_data', 'arpu_increase_due_to_the_upgrade_voice']]
                                .sum().reset_index())

    # Number of months per year to compute the opex costs
    df_increase_arpu_due_to_the_upgrade['week_date'] = pd.to_datetime(df_increase_arpu_due_to_the_upgrade['week_date'])
    df_increase_arpu_due_to_the_upgrade['month_period'] = (df_increase_arpu_due_to_the_upgrade['week_date']
    .apply(get_month_year_period)
)

    df_months_per_year = df_increase_arpu_due_to_the_upgrade[['site_id', 'year', 'month_period']]
    df_months_per_year = df_months_per_year.drop_duplicates()
    df_months_per_year = df_months_per_year.groupby(['site_id', 'year'])['month_period'].count().reset_index()

    df_months_per_year.columns = ['site_id', 'year', 'number_of_months_per_year']
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_months_per_year,on=['site_id', 'year'], how='left')

    # Compute the annual revenues by service
    revenues_per_unit_traffic['revenue_data'] = revenues_per_unit_traffic.revenue_data_mobile
    # + revenues_per_unit_traffic.revenue_data_box
    df_annual_revenues_by_service = (
        revenues_per_unit_traffic.groupby(['site_id'])[['revenue_data', 'revenue_voice_mobile']].sum()
        .reset_index())
    df_annual_revenues_by_service.columns = ['site_id', 'annual_revenues_data_traffic', 'annual_revenues_voice_traffic']

    # Merge with the increase in arpu due to the upgrade
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_annual_revenues_by_service, on='site_id', how='left')

    df_increase_arpu_by_year['increase_yearly_data_revenues'] = (
        df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_data'] /
        df_increase_arpu_by_year['annual_revenues_data_traffic'])

    df_increase_arpu_by_year['increase_yearly_voice_revenues'] = (
        df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_voice'] /
        df_increase_arpu_by_year['annual_revenues_voice_traffic'])

    df_increase_arpu_by_year['increase_yearly_revenues'] = (
        (df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_data'] +
        df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_voice']) /
        (df_increase_arpu_by_year['annual_revenues_data_traffic'] +
        df_increase_arpu_by_year['annual_revenues_voice_traffic']))

    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_margin_per_site[['site_id', 'site_margin_yearly']],
                                                              on='site_id', how='left')

    # Compute the increase in yearly data and voice margin due to the upgrade
    df_increase_arpu_by_year['increase_yearly_margin_due_to_the_upgrade'] = (
        df_increase_arpu_by_year['increase_yearly_revenues'] *
        df_increase_arpu_by_year['site_margin_yearly'])

    print("df_increase_arpu_by_year shape after: ", df_increase_arpu_by_year.shape)
    df_increase_arpu_by_year.to_parquet(increase_arpu_by_year_data_output.path)

    print("columns", df_increase_arpu_by_year.info())

    pandas_gbq.to_gbq(df_increase_arpu_by_year, increase_arpu_by_year_table_id, project_id=project_id,
                      location=location, if_exists='replace')
