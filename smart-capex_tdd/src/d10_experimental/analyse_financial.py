import os

import numpy as np
import pandas as pd

from src.d00_conf.conf import conf, conf_loader

conf_loader('oma.json')

use_case = conf['USE_CASE']
print(use_case)

TMP_SUFFIX  = '_from_capacity.csv'
def get_revenue_increase_voice_data_yearly(df_increase_arpu_due_to_the_upgrade):
    df_revenue_increase_voice_data_yearly = df_increase_arpu_due_to_the_upgrade.groupby(
        ['site_id', 'bands_upgraded', 'year']).agg({
        'traffic_increase_due_to_the_upgrade_data_mobile': sum,
        'traffic_increase_due_to_the_upgrade_data_box': sum,
        'traffic_increase_due_to_the_upgrade_voice': sum,
        'unit_price_data_box_with_the_decrease': np.mean,
        'unit_price_data_mobile_with_the_decrease': np.mean,
        'unit_price_voice_with_the_decrease': np.mean}).reset_index()

    df_revenue_increase_voice_data_yearly.rename(columns={
        'traffic_increase_due_to_the_upgrade_data_mobile': 'increase_data_traffic_Year',
        'traffic_increase_due_to_the_upgrade_data_box': 'increase_data_box_traffic_Year',
        'traffic_increase_due_to_the_upgrade_voice': 'increase_voice_traffic_Year',
        'unit_price_data_box_with_the_decrease': 'Price_Mo_data_box_year',
        'unit_price_data_mobile_with_the_decrease': 'Price_Mo_data_year',
        'unit_price_voice_with_the_decrease': 'Price_min_voice_year'
    }, inplace=True)

    df_revenue_increase_voice_data_yearly['increase_annual_Revenue_data_box'] = (
            df_revenue_increase_voice_data_yearly.increase_data_box_traffic_Year * \
            df_revenue_increase_voice_data_yearly.Price_Mo_data_box_year * 1024)
    df_revenue_increase_voice_data_yearly['increase_annual_Revenue_data'] = (
            df_revenue_increase_voice_data_yearly.increase_data_traffic_Year * \
            df_revenue_increase_voice_data_yearly.Price_Mo_data_year * 1024)
    df_revenue_increase_voice_data_yearly['increase_annual_Revenue_voice'] = \
    df_revenue_increase_voice_data_yearly.increase_voice_traffic_Year * \
    df_revenue_increase_voice_data_yearly.Price_min_voice_year * 1000 * 60

    return df_revenue_increase_voice_data_yearly


# def get_voice_data_annual_revenue_and_client_opex(df_revenues_per_unit_traffic):
#     df_voice_data_annual_revenue = df_revenues_per_unit_traffic.groupby(['site_id']).agg({
#         'revenues_voice_traffic': np.mean,
#         'revenues_data_traffic': np.mean,
#         'voice_opex_site': np.mean,
#         'data_opex_site': np.mean
#     }).reset_index()
#
#     # Monthly --> yearly
#     df_voice_data_annual_revenue.revenues_voice_traffic =
#     df_voice_data_annual_revenue.revenues_voice_traffic * 12
#     df_voice_data_annual_revenue.revenues_data_traffic =
#     df_voice_data_annual_revenue.revenues_data_traffic * 12
#     df_voice_data_annual_revenue.voice_opex_site =
#     df_voice_data_annual_revenue.voice_opex_site * 12
#     df_voice_data_annual_revenue.data_opex_site = df_voice_data_annual_revenue.data_opex_site * 12
#
#     df_voice_data_annual_revenue.rename(columns={
#         'revenues_voice_traffic': 'Annual_Revenue_voice',
#         'revenues_data_traffic': 'Annual_Revenue_data',
#         'voice_opex_site': 'Annual_Client_Opex_data',
#         'data_opex_site': 'Annual_Client_Opex_voice'
#     }, inplace=True)
#     return df_voice_data_annual_revenue


def get_annual_revenue_and_opex(df_margin_per_site_details):
    # Try to rename count_interco in opex colients
    df_margin_per_site_details.rename(columns={
        'cout_interco': 'opex_clients',
        'cluster_key': 'site_id',
    }, inplace=True)

    df_margin_per_site_details = df_margin_per_site_details.groupby(['site_id']).agg({
        'revenue_voice_mobile': sum,
        'revenue_data_mobile': sum,
        'revenue_total_box': sum,
        'revenue_total_site': sum,
        'opex_clients': sum}).reset_index(drop=False)

    df_margin_per_site_details.rename(columns={
        'revenue_voice_mobile': 'Annual_Revenue_voice',
        'revenue_data_mobile': 'Annual_Revenue_data',
        'revenue_total_box': 'Annual_Revenue_Box',
        'opex_clients': 'Annual_Client_Opex'
    }, inplace=True)
    return df_margin_per_site_details


def get_all_info_for_global_financial_analysis():
    # Yearly Revenue Increase on service voice & data
    # df_increase_arpu_due_to_the_upgrade = pd.read_csv(
    #    "/data/OMA/02_intermediate/tech_to_eco/df_increase_arpu_due_to_the_upgrade_25012024.csv",
    #    sep="|")
    df_increase_arpu_due_to_the_upgrade = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_') +
        conf['USE_CASE'] + TMP_SUFFIX, sep='|')

    df_increase_arpu_due_to_the_upgrade_site = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'df_increase_arpu_due_to_the_upgrade_site_') + conf['USE_CASE']
        + TMP_SUFFIX, sep='|')
    print(df_increase_arpu_due_to_the_upgrade_site)


    df_revenue_increase_voice_data_yearly = get_revenue_increase_voice_data_yearly(
        df_increase_arpu_due_to_the_upgrade)

    # NPV + Cashflows #
    year0 = 2023
    df_revenue_increase_voice_data_yearly['num_year'] = df_revenue_increase_voice_data_yearly[
                                                            'year'] - year0
    df_site_yearly_bp = (
        df_revenue_increase_voice_data_yearly.pivot_table(index=['site_id', 'bands_upgraded'],
                                                          columns=['num_year'],
                                                          values=[
                                                              'increase_data_traffic_Year',
                                                              'Price_Mo_data_year',
                                                              'increase_data_box_traffic_Year',
                                                              'Price_Mo_data_box_year',
                                                              'increase_voice_traffic_Year',
                                                              'Price_min_voice_year'

                                                            # 'increase_annual_Revenue_data_box',
                                                            # 'increase_annual_Revenue_data_mobile',
                                                            # 'increase_annual_Revenue_voice'
                                                          ])
        .add_prefix('')
        .reset_index())

    df_site_yearly_bp.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in
                                 df_site_yearly_bp.columns]
    # df_revenues_per_unit_traffic = pd.read_csv(
    #    '/data/OMA/02_intermediate/tech_to_eco/revenues_per_unit_traffic_25012024.csv', sep='|')
    df_revenues_per_unit_traffic = pd.read_csv(
        os.path.join(conf['PATH']['INTERMEDIATE_DATA'], 'tech_to_eco',
                     'revenues_per_unit_traffic_') +
        conf['USE_CASE'] + TMP_SUFFIX, sep='|')
    df_margin_per_site_details = get_annual_revenue_and_opex(df_revenues_per_unit_traffic)
    df_site_yearly_bp = df_site_yearly_bp.merge(df_margin_per_site_details, on=['site_id'],
                                                how='left')



    df_npv_of_the_upgrade = pd.read_csv(
        os.path.join(conf['PATH']['MODELS_OUTPUT'], 'final_npv', '20240624_105338',
                     'final_npv_of_the_upgrade_') +
        conf['USE_CASE'] + TMP_SUFFIX, sep='|')
    # Rename bands upgraded as cell_band for the merge
    df_site_yearly_bp.rename(columns={"bands_upgraded": "cell_band"}, inplace=True)
    df_npv_of_the_upgrade["cell_band"] = "densification"
    df_npv_of_the_upgrade['site_id'] = (
        df_npv_of_the_upgrade['site_id'].str.split('_').str[:2].str.join('_'))
    # df_site_yearly_bp = df_site_yearly_bp.merge(df_npv_of_the_upgrade,
    # on=['site_id', 'cell_band'], how='left')
    df_site_yearly_bp = df_site_yearly_bp.merge(df_npv_of_the_upgrade, on=['site_id', 'cell_band'],
                                                how='inner')

    df_site_yearly_bp.rename(columns={"Annual_Client_Opex": "Annual_Client_Opex_data",
                                      'opex_cost_year_0.0': 'opex_cost_year_0',
                                      'opex_cost_year_1.0': 'opex_cost_year_1',
                                      'opex_cost_year_2.0': 'opex_cost_year_2',
                                      'opex_cost_year_3.0': 'opex_cost_year_3',
                                      'opex_cost_year_4.0': 'opex_cost_year_4',
                                      'opex_cost_year_5.0': 'opex_cost_year_5'}, inplace=True)

    df_site_yearly_bp = df_site_yearly_bp[['site_id', 'cell_band',
                                           'increase_data_traffic_Year_1',
                                           'increase_data_traffic_Year_2',
                                           'increase_data_traffic_Year_3',
                                           'increase_data_traffic_Year_4',
                                           'increase_data_traffic_Year_5',
                                           'increase_voice_traffic_Year_1',
                                           'increase_voice_traffic_Year_2',
                                           'increase_voice_traffic_Year_3',
                                           'increase_voice_traffic_Year_4',
                                           'increase_voice_traffic_Year_5',
                                           'Price_min_voice_year_1', 'Price_min_voice_year_2',
                                           'Price_min_voice_year_3',
                                           'Price_min_voice_year_4', 'Price_min_voice_year_5',
                                           'Price_Mo_data_year_1', 'Price_Mo_data_year_2',
                                           'Price_Mo_data_year_3',
                                           'Price_Mo_data_year_4', 'Price_Mo_data_year_5',
                                           'Annual_Revenue_voice', 'Annual_Revenue_data',
                                           'Annual_Client_Opex_data',
                                           'cash_flow_year_0', 'cash_flow_year_1',
                                           'cash_flow_year_2',
                                           'cash_flow_year_3', 'cash_flow_year_4',
                                           'cash_flow_year_5',
                                           'opex_cost_year_0', 'opex_cost_year_1',
                                           'opex_cost_year_2',
                                           'opex_cost_year_3', 'opex_cost_year_4',
                                           'opex_cost_year_5',
                                           'NPV', 'Annual_Revenue_Box',
                                           'increase_data_box_traffic_Year_1',
                                           'increase_data_box_traffic_Year_2',
                                           'increase_data_box_traffic_Year_3',
                                           'increase_data_box_traffic_Year_4',
                                           'increase_data_box_traffic_Year_5',
                                           'Price_Mo_data_box_year_1', 'Price_Mo_data_box_year_2',
                                           'Price_Mo_data_box_year_3', 'Price_Mo_data_box_year_4',
                                           'Price_Mo_data_box_year_5'
                                           ]]

    # add  columns
    missing_cols = [
        'increase_OM_transaction_Year_1', 'increase_OM_transaction_Year_2',
        'increase_OM_transaction_Year_3',
        'increase_OM_transaction_Year_4', 'increase_OM_transaction_Year_5',
        'increase_OM_transaction_Year_6',
        'Fees_OM_year_1', 'Fees_OM_year_2', 'Fees_OM_year_3', 'Fees_OM_year_4', 'Fees_OM_year_5',
        'Fees_OM_year_6',
        'Annual_Revenue_OM', 'Annual_Client_Opex_OM', 'Annual_Client_Opex_voice',
        'Price_Mo_data_year_6',
        'Price_min_voice_year_6', 'cash_flow_year_6', 'opex_cost_year_6',
        'increase_data_traffic_Year_6',
        'increase_voice_traffic_Year_6', 'increase_data_box_traffic_Year_6',
        'Price_Mo_data_box_year_6',]
        #increase_voice_traffic_Year_3', 'increase_voice_traffic_Year_4',
        #increase_voice_traffic_Year_5', 'increase_voice_traffic_Year_1',
        #increase_voice_traffic_Year_2']
    df_site_yearly_bp = df_site_yearly_bp.assign(**dict.fromkeys(missing_cols, 0))

    df_site_yearly_bp = df_site_yearly_bp[[
        'site_id', 'cell_band', 'increase_data_traffic_Year_1', 'increase_data_traffic_Year_2',
        'increase_data_traffic_Year_3',
        'increase_data_traffic_Year_4', 'increase_data_traffic_Year_5',
        'increase_data_traffic_Year_6',
        'increase_voice_traffic_Year_1',
        'increase_voice_traffic_Year_2', 'increase_voice_traffic_Year_3',
        'increase_voice_traffic_Year_4',
        'increase_voice_traffic_Year_5', 'increase_voice_traffic_Year_6',
        'increase_OM_transaction_Year_1',
        'increase_OM_transaction_Year_2',
        'increase_OM_transaction_Year_3', 'increase_OM_transaction_Year_4',
        'increase_OM_transaction_Year_5',
        'increase_OM_transaction_Year_6',
        'Price_min_voice_year_1', 'Price_min_voice_year_2', 'Price_min_voice_year_3',
        'Price_min_voice_year_4',
        'Price_min_voice_year_5', 'Price_min_voice_year_6', 'Price_Mo_data_year_1',
        'Price_Mo_data_year_2',
        'Price_Mo_data_year_3', 'Price_Mo_data_year_4',
        'Price_Mo_data_year_5', 'Price_Mo_data_year_6', 'Fees_OM_year_1', 'Fees_OM_year_2',
        'Fees_OM_year_3',
        'Fees_OM_year_4', 'Fees_OM_year_5', 'Fees_OM_year_6',
        'Annual_Revenue_voice', 'Annual_Revenue_data', 'Annual_Revenue_OM',
        'Annual_Client_Opex_voice',
        'Annual_Client_Opex_data',
        'Annual_Client_Opex_OM', 'cash_flow_year_0', 'cash_flow_year_1', 'cash_flow_year_2',
        'cash_flow_year_3',
        'cash_flow_year_4',
        'cash_flow_year_5', 'cash_flow_year_6', 'opex_cost_year_0', 'opex_cost_year_1',
        'opex_cost_year_2',
        'opex_cost_year_3', 'opex_cost_year_4',
        'opex_cost_year_5', 'opex_cost_year_6', 'NPV',
        'Annual_Revenue_Box', 'increase_data_box_traffic_Year_1',
        'increase_data_box_traffic_Year_2', 'increase_data_box_traffic_Year_3',
        'increase_data_box_traffic_Year_4', 'increase_data_box_traffic_Year_5',
        'increase_data_box_traffic_Year_6',
        'Price_Mo_data_box_year_1', 'Price_Mo_data_box_year_2',
        'Price_Mo_data_box_year_3', 'Price_Mo_data_box_year_4',
        'Price_Mo_data_box_year_5', 'Price_Mo_data_box_year_6'
    ]]

    df_site_yearly_bp.to_csv(
        'data/OMA/TDD/09_reporting/df_npv_validation_DVM_increase_year_' + conf[
            'USE_CASE'] + '.csv', sep='|',
        index=False)

    df_increase_per_month = df_increase_arpu_due_to_the_upgrade.copy()

    df_increase_per_month_agg = df_increase_per_month.groupby(
        ['site_id', 'bands_upgraded', 'year']).agg(
        {'arpu_increase_due_to_the_upgrade_data_box': sum,
         'arpu_increase_due_to_the_upgrade_data_mobile': sum,
       'arpu_increase_due_to_the_upgrade_voice': sum}).reset_index()
    df_increase_per_month_agg.columns = ['site_id', 'bands_upgraded', 'year',
                                         'arpu_increase_due_to_the_upgrade_data_box_Year',
                                         'arpu_increase_due_to_the_upgrade_data_Year',
                                         'arpu_increase_due_to_the_upgrade_voice_Year']

    year0 = 2023
    df_increase_per_month_agg['num_year'] = df_increase_per_month_agg['year'] - year0
    df_increase_per_month_agg_pivot = (
        df_increase_per_month_agg.pivot_table(index=['site_id', 'bands_upgraded'],
                                              columns=['num_year'], values=[
                'arpu_increase_due_to_the_upgrade_data_Year',
                'arpu_increase_due_to_the_upgrade_data_box_Year',
                'arpu_increase_due_to_the_upgrade_voice_Year',
            ])
        .add_prefix('')
        .reset_index())

    df_increase_per_month_agg_pivot.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for
                                               col in
                                               df_increase_per_month_agg_pivot.columns]
    df_increase_per_month_agg_pivot.rename(columns={"bands_upgraded": "cell_band"}, inplace=True)

    df_site_yearly_bp_merge = df_site_yearly_bp.merge(df_increase_per_month_agg_pivot, how='left',
                                                      on=['site_id', 'cell_band'])
    missing_cols = ['arpu_increase_due_to_the_upgrade_voice_Year_6',
                    'arpu_increase_due_to_the_upgrade_data_box_Year_6',
                    'arpu_increase_due_to_the_upgrade_data_Year_6',]
                    #'arpu_increase_due_to_the_upgrade_voice_Year_5',]
                    #'arpu_increase_due_to_the_upgrade_voice_Year_1',
                    #'arpu_increase_due_to_the_upgrade_voice_Year_4',
                    #'arpu_increase_due_to_the_upgrade_voice_Year_3',
                    #'arpu_increase_due_to_the_upgrade_voice_Year_2']
    df_site_yearly_bp_merge = df_site_yearly_bp_merge.assign(**dict.fromkeys(missing_cols,
                                                                             0))
    df_site_yearly_bp_merge = df_site_yearly_bp_merge[[
        'site_id', 'cell_band', 'increase_data_traffic_Year_1', 'increase_data_traffic_Year_2',
        'increase_data_traffic_Year_3',
        'increase_data_traffic_Year_4', 'increase_data_traffic_Year_5',
        'increase_data_traffic_Year_6',
        'increase_voice_traffic_Year_1',
        'increase_voice_traffic_Year_2', 'increase_voice_traffic_Year_3',
        'increase_voice_traffic_Year_4',
        'increase_voice_traffic_Year_5', 'increase_voice_traffic_Year_6',
        'increase_OM_transaction_Year_1',
        'increase_OM_transaction_Year_2',
        'increase_OM_transaction_Year_3', 'increase_OM_transaction_Year_4',
        'increase_OM_transaction_Year_5',
        'increase_OM_transaction_Year_6',
        'Price_min_voice_year_1', 'Price_min_voice_year_2', 'Price_min_voice_year_3',
        'Price_min_voice_year_4',
        'Price_min_voice_year_5', 'Price_min_voice_year_6', 'Price_Mo_data_year_1',
        'Price_Mo_data_year_2',
        'Price_Mo_data_year_3', 'Price_Mo_data_year_4',
        'Price_Mo_data_year_5', 'Price_Mo_data_year_6', 'Fees_OM_year_1', 'Fees_OM_year_2',
        'Fees_OM_year_3',
        'Fees_OM_year_4', 'Fees_OM_year_5', 'Fees_OM_year_6',
        'Annual_Revenue_voice', 'Annual_Revenue_data', 'Annual_Revenue_OM',
        'Annual_Client_Opex_voice',
        'Annual_Client_Opex_data',
        'Annual_Client_Opex_OM', 'cash_flow_year_0', 'cash_flow_year_1', 'cash_flow_year_2',
        'cash_flow_year_3',
        'cash_flow_year_4',
        'cash_flow_year_5', 'cash_flow_year_6', 'opex_cost_year_0', 'opex_cost_year_1',
        'opex_cost_year_2',
        'opex_cost_year_3', 'opex_cost_year_4',
        'opex_cost_year_5', 'opex_cost_year_6', 'NPV',
        'arpu_increase_due_to_the_upgrade_data_Year_1',
        'arpu_increase_due_to_the_upgrade_data_Year_2',
        'arpu_increase_due_to_the_upgrade_data_Year_3',
        'arpu_increase_due_to_the_upgrade_data_Year_4',
        'arpu_increase_due_to_the_upgrade_data_Year_5',
        'arpu_increase_due_to_the_upgrade_data_Year_6',
         'arpu_increase_due_to_the_upgrade_voice_Year_1',
         'arpu_increase_due_to_the_upgrade_voice_Year_2',
         'arpu_increase_due_to_the_upgrade_voice_Year_3',
         'arpu_increase_due_to_the_upgrade_voice_Year_4',
         'arpu_increase_due_to_the_upgrade_voice_Year_5',
        'arpu_increase_due_to_the_upgrade_voice_Year_6',
        'Annual_Revenue_Box', 'increase_data_box_traffic_Year_1',
        'increase_data_box_traffic_Year_2', 'increase_data_box_traffic_Year_3',
        'increase_data_box_traffic_Year_4', 'increase_data_box_traffic_Year_5',
        'increase_data_box_traffic_Year_6',
        'arpu_increase_due_to_the_upgrade_data_box_Year_1',
        'arpu_increase_due_to_the_upgrade_data_box_Year_2',
        'arpu_increase_due_to_the_upgrade_data_box_Year_3',
        'arpu_increase_due_to_the_upgrade_data_box_Year_4',
        'arpu_increase_due_to_the_upgrade_data_box_Year_5',
        'arpu_increase_due_to_the_upgrade_data_box_Year_6',
        'Price_Mo_data_box_year_1', 'Price_Mo_data_box_year_2',
        'Price_Mo_data_box_year_3', 'Price_Mo_data_box_year_4',
        'Price_Mo_data_box_year_5', 'Price_Mo_data_box_year_6'
    ]]

    # Sortie utilise si prix fixe par an
    # df_site_yearly_bp_merge.to_csv(
    #    'data/OMA/09_reporting/df_npv_validation_DVM_with_increase_25012024.csv', sep='|',
    #    index=False)
    ## Sortie si prix par mois (plusieurs prix par an)
    # df_increase_per_month_agg_pivot.to_csv(
    #    'data/OMA/09_reporting/df_npv_validation_DVM_increase_revenue_25012024.csv', sep='|',
    #    index=False)
    df_site_yearly_bp_merge.to_csv(
        'data/OMA/TDD/09_reporting/df_npv_validation_DVM_with_increase_' + conf[
            'USE_CASE'] + '.csv', sep='|',
        index=False
    )
    df_increase_per_month_agg_pivot.to_csv(
        'data/OMA/TDD/09_reporting/df_npv_validation_DVM_increase_revenue_' + conf[
            'USE_CASE'] + '.csv', sep='|',
        index=False
    )
    return df_site_yearly_bp


if __name__ == "__main__":
    df_sites_yearly_bp = get_all_info_for_global_financial_analysis()
