from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_increase_cash_flow(project_id: str,
                               location: str,
                               cash_flow_table_id: str,
                               time_to_compute_npv: int,
                               opex_data_input: Input[Dataset],
                               capex_data_input: Input[Dataset],
                               increase_in_margin_due_to_the_upgrade_data_input: Input[Dataset],
                               cash_flow_data_output: Output[Dataset]):

    """It computes the increase in cash flow resulting from a site upgrade
    
    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        cash_flow_table_id (str): It holds the resource name on BigQuery
        time_to_compute_npv (int): It holds the number of years to compute the Net Present Value (NPV).
        opex_data_input (Input[Dataset]): It holds opex data
        capex_data_input (Input[Dataset]): It holds capex data
        increase_in_margin_due_to_the_upgrade_data_input (Input[Dataset]): It holds computed yearly increase in arpu
        cash_flow_data_output (Output[Dataset]): It holds the computed cash flow data.

    Returns:
        cash_flow_data_output (Output[Dataset]): It holds the computed cash flow data.
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    def get_opex_cost(row):
        opex_column = 'yearly_rural_opex' if row['site_area'] == 'RURAL' else 'yearly_urban_opex'
        opex_cost = df_opex.loc[df_opex['bands_upgraded'] == row['bands_upgraded'], opex_column]

        return opex_cost

    # Load Data
    df_capex = pd.read_parquet(capex_data_input.path)
    df_opex = pd.read_parquet(opex_data_input.path)
    df_increase_in_margin_due_to_the_upgrade = pd.read_parquet(increase_in_margin_due_to_the_upgrade_data_input.path)

    # df_opex = df_opex[['bands', 'opex_cost']]
    df_opex = df_opex.rename(columns={'bands':'bands_upgraded'})

    # Minimum year by site
    df_min_year = df_increase_in_margin_due_to_the_upgrade.groupby(['site_id'])['year'].min().reset_index()
    df_min_year.columns = ['site_id', 'min_year']
    df_increase_in_margin_due_to_the_upgrade = (
        df_increase_in_margin_due_to_the_upgrade.merge(df_min_year, on='site_id', how='left'))

    # Compute cash flow year
    #-------
    df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = (
        df_increase_in_margin_due_to_the_upgrade['year'].apply(int) -
        df_increase_in_margin_due_to_the_upgrade['min_year'].apply(int))

    df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = (
        df_increase_in_margin_due_to_the_upgrade['cash_flow_year'].astype(int))

    # Limitating NPV to number of years define in conf file
    df_increase_in_margin_due_to_the_upgrade = (
        df_increase_in_margin_due_to_the_upgrade[df_increase_in_margin_due_to_the_upgrade["cash_flow_year"]
                                                 <= (time_to_compute_npv + 1)])

    df_increase_in_margin_due_to_the_upgrade.year = df_increase_in_margin_due_to_the_upgrade.year.astype(int)

    # Merge with site opex costs
    # df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(df_opex,
    #                                            on='bands_upgraded', how='left')
    df_increase_in_margin_due_to_the_upgrade['opex_costs'] = (df_increase_in_margin_due_to_the_upgrade
                                                              .apply(get_opex_cost, axis=1))
    # Merge with the band
    df_increase_in_margin_due_to_the_upgrade.rename(columns={'bands_upgraded': 'cell_band'}, inplace=True)

    # Join the increase in data and voice margin
    df_increase_in_margin_due_to_the_upgrade.opex_costs.fillna(0, inplace=True)

    # df_increase_in_margin_due_to_the_upgrade.opex_costs = df_increase_in_margin_
    # due_to_the_upgrade.opex_costs.apply(
    #     lambda x: float(x.split()[0].replace(',', '')) if type(x) == str else x)
    df_increase_in_margin_due_to_the_upgrade.opex_costs = (
        df_increase_in_margin_due_to_the_upgrade.opex_costs *
        (df_increase_in_margin_due_to_the_upgrade.number_of_months_per_year / 12))
    # Compute the cash flow increment by substracting the opex costs
    df_increase_in_margin_due_to_the_upgrade['increase_cash_flow_due_to_the_upgrade'] = (
        df_increase_in_margin_due_to_the_upgrade['increase_yearly_margin_due_to_the_upgrade'] -
        df_increase_in_margin_due_to_the_upgrade['opex_costs'])

    # Capex is in the cash flow of year 0
    df_capex.columns = ['bands', 'capex_cost']
    df_capex['capex_cost'] = df_capex['capex_cost'] * 1000000
    df_capex['cash_flow_year'] = 0

    df_capex['increase_cash_flow_due_to_the_upgrade'] = -df_capex['capex_cost']


    df_capex_site = df_increase_in_margin_due_to_the_upgrade[['site_id', 'cell_band', 'site_area']].drop_duplicates()
    df_capex_site = df_capex_site.merge(df_capex, right_on=['bands'], left_on=['cell_band'], how='left')

    # Add Capex for the upgrade not matching the capex_file
    # To do so we first calculate a mean capex per technology
    # Save the cash flow results
    df_final = pd.concat([df_increase_in_margin_due_to_the_upgrade, df_capex_site])
    # df_final.drop(columns=['Tech'], inplace=True)
    df_final.reset_index(inplace=True)
    df_final.drop('index', axis=1, inplace=True)
    # Transforming capex in Euros to DH
    # df_final.increase_cash_flow_due_to_the_upgrade =
    # df_final.increase_cash_flow_due_to_the_upgrade * pipeline_config['NPV'][
    #     'EURO_TO_DH'

    print("df_final shape: ", df_final.shape)
    df_final.to_parquet(cash_flow_data_output.path)

    print("columns", df_final.info())

    print("bands", df_final["bands"].value_counts())

    pandas_gbq.to_gbq(df_final, cash_flow_table_id, project_id=project_id,
                      location=location, if_exists='replace')
