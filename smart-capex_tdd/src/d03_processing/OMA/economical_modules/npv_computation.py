"""Module to compute NPV"""
import os

import numpy as np
import numpy_financial as npf
import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import write_csv_sql, npv_since_2nd_years, add_logging_info


@add_logging_info
def compute_increase_cash_flow(df_increase_in_margin_due_to_the_upgrade):
    """
    The compute_increase_cash_flow function calculates the increase in cash flow due to an upgrade
    by processing a DataFrame. It computes the minimum year per site, calculates the cash flow year,
    limits the NPV to a defined number of years, merges with site OPEX costs,
    and computes the cash flow increment. Finally, it saves the results to a CSV file.

    Parameters
    ----------
    df_increase_in_margin_due_to_the_upgrade: pd.DataFrame

    Returns
    -------
    df_final : pd.DataFrame
        A pandas DataFrame containing the computed increase in cash flow due to the upgrade.

    """
    # Minimum year by site
    df_min_year = df_increase_in_margin_due_to_the_upgrade.groupby(
        ['cluster_key'])['year'].min().reset_index()
    df_min_year.columns = ['cluster_key', 'min_year']
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(
        df_min_year, on='cluster_key', how='left')
    # Compute cash flow year
    df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = (
            (df_increase_in_margin_due_to_the_upgrade['year'].apply(int) -
             df_increase_in_margin_due_to_the_upgrade['min_year'].apply(int)) + 1)
    # Limitating NPV to number of years define in conf file
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade[
        df_increase_in_margin_due_to_the_upgrade.cash_flow_year <=
        (conf['NPV']['TIME_TO_COMPUTE_NPV'] + 1)]

    # Merge with site opex costs
    df_increase_in_margin_due_to_the_upgrade.year = (
        df_increase_in_margin_due_to_the_upgrade.year.astype(int))

    # Opex 7800 Mois (Info de pierre) # 7800 / 7200 ?
    df_increase_in_margin_due_to_the_upgrade["opex_costs"] = (
            conf['NPV']['OPEX_COSTS'] * conf['NPV']['NB_MONTHS'])

    df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = (
        df_increase_in_margin_due_to_the_upgrade['cash_flow_year'].astype(int))
    df_increase_in_margin_due_to_the_upgrade.rename(columns={'bands_upgraded': 'cell_band'},
                                                    inplace=True)
    # Join the increase in data and voice margin
    df_increase_in_margin_due_to_the_upgrade.opex_costs.fillna(0, inplace=True)
    df_increase_in_margin_due_to_the_upgrade.opex_costs = (
            df_increase_in_margin_due_to_the_upgrade.opex_costs *
            (df_increase_in_margin_due_to_the_upgrade.number_of_months_per_year / 12))
    # Compute the cash flow increment by substracting the opex costs
    df_increase_in_margin_due_to_the_upgrade['increase_cash_flow_due_to_the_upgrade'] = \
        df_increase_in_margin_due_to_the_upgrade['increase_yearly_margin_due_to_the_upgrade'] - \
        df_increase_in_margin_due_to_the_upgrade['opex_costs']


    df_capex_site = df_increase_in_margin_due_to_the_upgrade[
        ['cluster_key']].drop_duplicates()

    # Capex valeur fixe intégré
    df_capex_site['increase_cash_flow_due_to_the_upgrade'] = conf['NPV']['CAPEX']
    df_capex_site['cash_flow_year'] = conf['NPV']['CASH_FLOW_YEAR_0']

    # Save the cash flow results
    df_final = pd.concat([df_increase_in_margin_due_to_the_upgrade, df_capex_site])
    df_final['cell_band'] = 'Densification'
    # df_final.drop(columns=['Tech'], inplace=True)
    df_final.reset_index(inplace=True)
    df_final.drop('index', axis=1, inplace=True)
    # Transforming capex in Euros to DH
    # df_final.increase_cash_flow_due_to_the_upgrade =
    # df_final.increase_cash_flow_due_to_the_upgrade * conf['NPV']['EURO_TO_DH']

    cash_flow_path = os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                  "increase_in_cash_flow_due_to_the_upgrade",
                                  conf['EXEC_TIME'])

    if not os.path.exists(cash_flow_path):
        os.makedirs(cash_flow_path)

    write_csv_sql(df_final, 'increase_in_cash_flow_due_to_the_upgrade_' + conf[
        "USE_CASE"] + '_from_capacity.csv',
                  path=cash_flow_path, sql=False, csv=True)
    return df_final


@add_logging_info
def compute_npv(df_cash_flow):
    """
    The compute_npv function calculates the Net Present Value (NPV) for a given cash flow DataFrame,
    reorganizes the dataset, computes additional financial metrics like EBITDA and IRR,
    and saves the final results to a CSV file.

    Parameters
    ----------
    df_cash_flow: pd.DataFrame
        A pandas DataFrame containing cash flow data with columns like cluster_key, cash_flow_year,
        increase_cash_flow_due_to_the_upgrade, and opex_costs.

    Returns
    -------
    df: pd.DataFrame
        A pandas DataFrame containing the final NPV, total opex, total revenue, EBITDA,
        and IRR for each site and cell band.

    """

    # Aux funtion to compute the NPV


    nb_year = df_cash_flow['cash_flow_year'].unique()
    df_cash_flow = df_cash_flow.rename(columns={'cluster_key': 'site_id'})
    nb_year = np.sort(nb_year)



    df_npv = compute_df_cash_flow_discount(df_cash_flow, nb_year)

    df_npv['NPV'] = df_npv['capex_cf_y1'] + df_npv['NPV_cf_y2']
    df_npv.drop(['capex_cf_y1', 'NPV_cf_y2'], axis=1, inplace=True)
    df_npv.columns = ['site_id', 'cell_band', 'NPV']

    # Reorganize the final dataset
    df, df_cash_flow_pv, df_opex_cost_year_pv, new_columns_names = reorganize_final_dataset(
        df_cash_flow)

    df = df.merge(df_cash_flow_pv,
                  on=['site_id', 'cell_band'],
                  how='left')

    df = df.merge(df_opex_cost_year_pv,
                  on=['site_id', 'cell_band'],
                  how='left')
    # Merge with the NVP
    df = df.merge(df_npv,
                  on=['site_id', 'cell_band'],
                  how='left')
    df['total_opex'] = df[[col for col in df.columns if col.startswith('opex')]].sum(axis=1)
    df['total_revenue'] = df[[col for col in df.columns if col.startswith('cash_flow')]].sum(
        axis=1) + df[
                              [col for col in df.columns if col.startswith('opex')]].sum(axis=1) - \
                          df['cash_flow_year_0']
    df['EBITDA_Value'] = df['total_revenue'] - df['total_opex']
    df['EBITDA'] = df['EBITDA_Value'] / df['total_revenue']

    df_irr = compute_df_irr(df, new_columns_names)

    df = df.merge(df_irr, on=['site_id', 'cell_band'], how='left')

    # Save the final results
    npv_path = os.path.join(conf['PATH']['MODELS_OUTPUT'],
                            "final_npv",
                            conf['EXEC_TIME'])
    if not os.path.exists(npv_path):
        os.makedirs(npv_path)

    write_csv_sql(df, 'final_npv_of_the_upgrade_' + conf["USE_CASE"] + '_from_capacity.csv',
                  path=npv_path, sql=False, csv=True)

    return df


def compute_npv_aux(df, wacc):
    """
    The compute_npv_aux function calculates the Net Present Value (NPV) of cash flows starting from
    the second year for a given DataFrame, using a specified Weighted Average Cost of Capital (WACC)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing cash flow data.
    wacc : float
        Weighted Average Cost of Capital as a percentage.
    Returns
    -------
    A float representing the Net Present Value (NPV) of the cash flows starting from the second year
    """
    df = df.reset_index()
    df = df[['cash_flow_year', 'increase_cash_flow_due_to_the_upgrade']].sort_values(
        by='cash_flow_year',
        ascending=True)
    return npv_since_2nd_years(values=df['increase_cash_flow_due_to_the_upgrade'].values,
                               rate=wacc / 100)

def compute_df_irr(df, new_columns_names):
    """
    The compute_df_irr function calculates the Internal Rate of Return (IRR) for each site and cell
    band in a DataFrame by reorganizing cash flow data and applying the IRR calculation

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing financial data.
    new_columns_names: list
        A list of new column names to be used in the IRR calculation
    Returns
    -------
    df_irr: pd.DataFrame
        A pandas DataFrame containing site_id, cell_band, and the calculated IRR
    """
    df_irr_columns = ['site_id', 'cell_band']
    df_irr_columns = df_irr_columns + new_columns_names
    df_irr_0_1 = df[df_irr_columns]
    df_irr_0_1['cash_flow_years_0_1'] = df_irr_0_1['cash_flow_year_0'] + df_irr_0_1[
        'cash_flow_year_1']
    df_irr_0_1.drop(['cash_flow_year_0', 'cash_flow_year_1'], axis=1, inplace=True)
    df_irr_0_1 = df_irr_0_1.reindex(
        columns=['site_id', 'cell_band', 'cash_flow_years_0_1'] + df_irr_columns[4:])
    irr_columns = df_irr_0_1.columns[2:]
    df_irr_0_1 = df_irr_0_1[df_irr_0_1.cash_flow_years_0_1.notna()]
    df_irr_0_1 = df_irr_0_1.fillna(0)
    df_irr_0_1['IRR'] = df_irr_0_1[irr_columns].apply(npf.irr)
    df_irr = df_irr_0_1[['site_id', 'cell_band', 'IRR']]
    return df_irr


def reorganize_final_dataset(df_cash_flow):
    """
    The reorganize_final_dataset function restructures the input DataFrame containing cash flow data
    by pivoting it to create separate DataFrames for cash flow and opex costs per year,
    and then merges these DataFrames into a final dataset

    Parameters
    ----------
    df_cash_flow: pd.DataFrame
        A pandas DataFrame containing columns like site_id, cell_band, cash_flow_year,
        increase_cash_flow_due_to_the_upgrade, and opex_costs.

    Returns
    -------
    df: pd.DataFrame
        A DataFrame with unique site_id and cell_band combinations.
    df_cash_flow: pd.DataFrame
        A pivoted DataFrame with cash flow values per year.
    df_opex_cost_year_pv: pd.DataFrame
        A pivoted DataFrame with opex costs per year.
    new_columns_names: list
        A list of new column names for the cash flow DataFrame.

    """
    df_cash_flow_pv = pd.pivot_table(df_cash_flow,
                                     values='increase_cash_flow_due_to_the_upgrade',
                                     index=['site_id', 'cell_band'],
                                     columns=['cash_flow_year'],
                                     aggfunc=np.sum)
    new_columns_names = []
    for i in df_cash_flow_pv.columns:
        new_columns_names.append('cash_flow_year_' + str(int(i)))
    df_cash_flow_pv.columns = new_columns_names
    df_cash_flow_pv.reset_index(inplace=True)
    df_opex_cost_year_pv = pd.pivot_table(df_cash_flow,
                                          values='opex_costs',
                                          index=['site_id', 'cell_band'],
                                          columns=['cash_flow_year'],
                                          aggfunc=np.sum)
    opex_columns_names = []
    for i in df_opex_cost_year_pv.columns:
        opex_columns_names.append('opex_cost_year_' + str(i))
    df_opex_cost_year_pv.columns = opex_columns_names
    df_opex_cost_year_pv.reset_index(inplace=True)
    df = df_cash_flow[['site_id', 'cell_band']].drop_duplicates()
    return df, df_cash_flow_pv, df_opex_cost_year_pv, new_columns_names


def compute_df_cash_flow_discount(df_cash_flow, nb_year):
    """
    The compute_df_cash_flow_discount function calculates the Net Present Value (NPV) of cash flows
    for each site and cell band, separating the first year's cash flow from subsequent years, and
    then merging the results into a final DataFrame.

    Parameters
    ----------
    df_cash_flow: pd.DataFrame
         pandas DataFrame containing cash flow data.
    nb_year: numpy.array
        A numpy array of unique years present in the cash flow data.
    Returns
    -------
    df_npv: pd.DataFrame
        A pandas DataFrame containing site_id, cell_band, capex_cf_y1, and NPV_cf_y2.
    """
    df_cash_flow_discount = df_cash_flow[~(df_cash_flow['cash_flow_year'].isin([nb_year[0]]))]
    df_cash_flow_no_discount = df_cash_flow[df_cash_flow['cash_flow_year'].isin([nb_year[0]])]
    df_cash_flow_no_discount_npv = df_cash_flow_no_discount.groupby(['site_id', 'cell_band'])[
        ['increase_cash_flow_due_to_the_upgrade']].sum().reset_index()
    df_cash_flow_no_discount_npv.columns = ['site_id', 'cell_band', 'capex_cf_y1']
    df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
        lambda x: compute_npv_aux(x, wacc=conf['NPV']['WACC'])).reset_index()
    df_cash_flow_discount_npv.columns = ['site_id', 'cell_band', 'NPV_cf_y2']
    df_npv = df_cash_flow_discount_npv.merge(df_cash_flow_no_discount_npv,
                                             on=['site_id', 'cell_band'],
                                             how='left')
    return df_npv
