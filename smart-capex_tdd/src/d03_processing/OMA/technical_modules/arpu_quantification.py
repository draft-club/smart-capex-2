"""Arpu Quantification"""
import os.path

import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import generate_bp_years, add_logging_info


@add_logging_info
def compute_revenues_per_site(df_traffic_weekly_kpis, df_opex, df_unit_prices,
                              df_weights, df_cluster_key, df_sites,
                              post_paid_pre_paid_region):
    """
    Function to compute revenue per site,  revenues are calculated for the neighbor

    Parameters
    ----------
    df_sites: pd.DataFrame
        Dataset with site's information
    df_traffic_weekly_kpis: pd.DataFrame
        Dataset with weekly kpi on historical data
    df_opex: pd.DataFrame
        Dataset with opex informations
    df_unit_prices: pd.DataFrame
    df_weights: pd.DataFrame
    df_cluster_key: pd.DataFrame

    Returns
    -------
    df_revenues_per_site: pd.DataFrame
    """

    # Prepare Dataset with ratio PostPaid PrePaid Per Region
    df_post_pre_paid_per_region = compute_post_pre_paid_per_region(df_cluster_key,
                                                                   df_sites,
                                                                   post_paid_pre_paid_region)
    # Renaming column
    df_opex.rename(columns={'site_name': 'site_id',
                            'cout_interco': 'opex_clients_interco'}, inplace=True)

    # df_cluster_key = prepare_neighbors_for_TDD()
    cols_to_keep = ['cell_name', 'cell_tech', 'cell_band', 'site_id', 'year', 'week',
                    'week_period', 'date', 'region', 'ville', 'province',
                    'total_data_traffic_dl_gb',
                    'total_voice_traffic_kerlangs']
    df_traffic_weekly_kpis = df_traffic_weekly_kpis[cols_to_keep]





    # Create year_month column and filter on specific year
    df_traffic_weekly_kpis["year_month"] = df_traffic_weekly_kpis["date"].apply(
        lambda x: str(x[:7]))
    df_traffic_weekly_kpis = (
        df_traffic_weekly_kpis)[df_traffic_weekly_kpis["year"] ==
                                conf['OSS_PREPROCESSING']['LAST_COMPLETE_YEAR_OF_OSS']]

    # Add weights informations
    df_traffic_weekly_kpis_merged_with_weights = \
        pd.merge(left=df_traffic_weekly_kpis, right=df_weights,
                 on=["year"], how="left")

    # Check 3G Cells and set weight for full mobile
    index_3g = df_traffic_weekly_kpis_merged_with_weights["cell_tech"] == "3G"
    df_traffic_weekly_kpis_merged_with_weights.loc[index_3g, "weight_box"] = 0
    df_traffic_weekly_kpis_merged_with_weights.loc[index_3g, "weight_mobile"] = 1

    df_traffic_weekly_kpis_merged_with_weights["total_data_box_traffic_dl_gb"] = \
        df_traffic_weekly_kpis_merged_with_weights["weight_box"] * \
        df_traffic_weekly_kpis_merged_with_weights["total_data_traffic_dl_gb"]
    df_traffic_weekly_kpis_merged_with_weights["total_data_mobile_traffic_dl_gb"] = \
        df_traffic_weekly_kpis_merged_with_weights["weight_mobile"] * \
        df_traffic_weekly_kpis_merged_with_weights["total_data_traffic_dl_gb"]

    # Get information of neighbour 1 & 2
    n1 = df_traffic_weekly_kpis_merged_with_weights.merge(df_cluster_key, how='left',
                                                          left_on='site_id',
                                                          right_on=['neighbour_1'])
    n2 = df_traffic_weekly_kpis_merged_with_weights.merge(df_cluster_key, how='left',
                                                          left_on='site_id',
                                                          right_on=['neighbour_2'])
    df_temp = pd.concat([n1, n2])



    # Test 07/06/2024
    df_traffic_sum_all_cells = df_temp.groupby(['cluster_key','week_period'])[[
        'total_data_box_traffic_dl_gb',
        'total_data_mobile_traffic_dl_gb',
        'total_voice_traffic_kerlangs']].sum().reset_index()
    print(df_traffic_sum_all_cells)

    df_traffic_avg_weekly_per_cluster = df_traffic_sum_all_cells.groupby(['cluster_key'])[[
        'total_data_box_traffic_dl_gb',
        'total_data_mobile_traffic_dl_gb',
        'total_voice_traffic_kerlangs']].mean().reset_index()

    df_traffic_yearly_kpis = df_traffic_avg_weekly_per_cluster.copy()
    for kpi in ['total_data_box_traffic_dl_gb', 'total_data_mobile_traffic_dl_gb',
                'total_voice_traffic_kerlangs']:
        df_traffic_yearly_kpis[kpi] = df_traffic_avg_weekly_per_cluster[kpi] * 52

    df_traffic_yearly_kpis = (
        pd.merge(left=df_traffic_yearly_kpis,
                 right=df_post_pre_paid_per_region[['cluster_key', 'neighbour_1', 'neighbour_2',
                                                    'region', 'ratio_prepaid', 'ratio_prepaid_inwi',
                                                    'ratio_prepaid_iam', 'ratio_postpaid',
                                                    'ratio_postpaid_inwi', 'ratio_postpaid_iam']],
                 on='cluster_key'))

    df_revenues_per_site = compute_revenue_and_opex(df_traffic_yearly_kpis, df_unit_prices)


    df_revenues_per_site.to_csv(
        os.path.join(conf["PATH"]["INTERMEDIATE_DATA"], 'revenues_per_unit_traffic.csv'),
        sep="|", index=False)
    return df_revenues_per_site, df_post_pre_paid_per_region


def compute_revenues_per_site_site(
        prediction_data_coverage,
        prediction_voice_coverage,
        df_opex,
        df_unit_prices,
        df_weights,
        df_cluster_key,
        df_sites,
        post_paid_pre_paid_region):
    """
    The compute_revenues_per_site_site function calculates the revenue and operational
    expenses (OPEX) for each site based on predicted data and voice coverage,
    site information, and various financial ratios.
    It merges multiple dataframes to compute these metrics and saves the results to a CSV file.

    Parameters
    ----------
    prediction_data_coverage: pd.DataFrame
         DataFrame containing predicted data traffic.
    prediction_voice_coverage : pd.DataFrame
         DataFrame containing predicted vocie traffic.
    df_opex : pd.DataFrame
        DataFrame containing operational expenses data.
    df_unit_prices : pd.DataFrame
        DataFrame containing unit prices for different services.
    df_weights : pd.DataFrame
        DataFrame containing weights for different traffic types.
    df_cluster_key : pd.DataFrame
        DataFrame containing cluster key information for sites.
    df_sites : pd.DataFrame
    post_paid_pre_paid_region : pd.DataFrame
         DataFrame containing post-paid and pre-paid ratios for each region.

    Returns
    -------
    df_revenues_per_site: pd.DataFrame
        DataFrame containing computed revenue and OPEX for each site.
    """

    df_post_pre_paid_per_region = compute_post_pre_paid_per_region(df_cluster_key,
                                                                   df_sites,
                                                                   post_paid_pre_paid_region)
    # Renaming column
    df_opex.rename(columns={'site_name': 'site_id',
                            'cout_interco': 'opex_clients_interco'}, inplace=True)

    prediction_data_coverage = (
        prediction_data_coverage.rename(columns={'predicted_traffic': 'total_data_traffic_dl_gb'}))

    df_traffic_weekly_kpis = prediction_data_coverage.copy()
    df_traffic_weekly_kpis['total_voice_traffic_kerlangs'] = (
        prediction_voice_coverage)['predicted_traffic']
    df_traffic_weekly_kpis['year'] = conf['OSS_PREPROCESSING']['LAST_COMPLETE_YEAR_OF_OSS']

    df_traffic_weekly_kpis_merged_with_weights = \
        pd.merge(left=df_traffic_weekly_kpis, right=df_weights,
                 on=["year"], how="left", validate="many_to_one")

    # Check 3G Cells and set weight for full mobile
    # index_3g = df_traffic_weekly_kpis_merged_with_weights["cell_tech"] == "3G"
    # df_traffic_weekly_kpis_merged_with_weights.loc[index_3g, "weight_box"] = 0
    # df_traffic_weekly_kpis_merged_with_weights.loc[index_3g, "weight_mobile"] = 1

    df_traffic_weekly_kpis_merged_with_weights["total_data_box_traffic_dl_gb"] = \
        df_traffic_weekly_kpis_merged_with_weights["weight_box"] * \
        df_traffic_weekly_kpis_merged_with_weights["total_data_traffic_dl_gb"]
    df_traffic_weekly_kpis_merged_with_weights["total_data_mobile_traffic_dl_gb"] = \
        df_traffic_weekly_kpis_merged_with_weights["weight_mobile"] * \
        df_traffic_weekly_kpis_merged_with_weights["total_data_traffic_dl_gb"]

    df_traffic_yearly_kpis = df_traffic_weekly_kpis_merged_with_weights.copy()
    df_traffic_yearly_kpis = pd.merge(left=df_traffic_yearly_kpis,
                                      right=df_cluster_key[['cluster_key', 'site_id']],
                                      on='site_id', how='left',validate="1:m")

    df_traffic_yearly_kpis = (
        pd.merge(left=df_traffic_yearly_kpis,
                 right=df_post_pre_paid_per_region[['cluster_key', 'neighbour_1', 'neighbour_2',
                                                    'region', 'ratio_prepaid', 'ratio_prepaid_inwi',
                                                    'ratio_prepaid_iam', 'ratio_postpaid',
                                                    'ratio_postpaid_inwi', 'ratio_postpaid_iam']],
                 on='cluster_key', how='left',validate="m:m"))

    df_revenues_per_site = compute_revenue_and_opex(df_traffic_yearly_kpis, df_unit_prices)

    df_revenues_per_site.to_csv(
        os.path.join(conf["PATH"]["INTERMEDIATE_DATA"], 'revenues_per_unit_traffic_site.csv'),
        sep="|", index=False)
    return df_revenues_per_site


def compute_revenue_and_opex(df_traffic_yearly_kpis, df_unit_prices):
    """
    The compute_revenue_and_opex function calculates the revenue and operational expenses (OPEX)
    for each site based on yearly traffic KPIs and unit prices.
    It computes various revenue streams (voice and data, both mobile and box) and OPEX components
    (commissions and interconnection costs) to provide a comprehensive financial overview per site.

    Parameters
    ----------
    df_traffic_yearly_kpis : pd.DataFrame
        dataFrame containing yearly traffic KPIs.
    df_unit_prices : pd.DataFrame
        DataFrame containing unit prices for different services.

    Returns
    -------
    df_revenues_per_site : pd.DataFrame
        DataFrame containing computed revenue and OPEX for each site.
    """
    df_temp_traffic_yearly_kpis_tmp = df_traffic_yearly_kpis.copy()
    # Compute KPis
    # Add post paid and prepaid
    df_temp_traffic_yearly_kpis_tmp["revenue_voice_mobile"] = (
            df_temp_traffic_yearly_kpis_tmp["total_voice_traffic_kerlangs"] *
            df_traffic_yearly_kpis['ratio_prepaid'] * conf['NPV']['ARPM_PREPAID'] +
            df_temp_traffic_yearly_kpis_tmp["total_voice_traffic_kerlangs"] *
            df_traffic_yearly_kpis['ratio_postpaid'] * conf['NPV']['ARPM_POSTPAID'] * 1000 * 60
    )
    df_temp_traffic_yearly_kpis_tmp["revenue_data_mobile"] = (
            df_temp_traffic_yearly_kpis_tmp["total_data_mobile_traffic_dl_gb"] *
            df_unit_prices[df_unit_prices["type"] == "arpg_data_mobile"][conf['OSS_PREPROCESSING']
            ['LAST_YEAR_OF_OSS']].values[0])
    df_temp_traffic_yearly_kpis_tmp["revenue_data_box"] = (
            df_temp_traffic_yearly_kpis_tmp["total_data_box_traffic_dl_gb"] *
            df_unit_prices[df_unit_prices["type"] == "arpg_data_box"][conf['OSS_PREPROCESSING']
            ['LAST_YEAR_OF_OSS']].values[0])
    df_revenues_per_site = df_temp_traffic_yearly_kpis_tmp.copy()
    df_revenues_per_site["revenue_total_mobile"] = df_revenues_per_site["revenue_voice_mobile"] + \
                                                   df_revenues_per_site["revenue_data_mobile"]
    df_revenues_per_site["revenue_total_box"] = df_revenues_per_site["revenue_data_box"]
    df_revenues_per_site["revenue_total_site"] = df_revenues_per_site["revenue_total_mobile"] \
                                                 + df_revenues_per_site["revenue_total_box"]
    df_revenues_per_site["opex_clients_commissions_data"] = (df_revenues_per_site[
                                                                 "revenue_data_mobile"] + \
                                                             df_revenues_per_site[
                                                                 "revenue_data_box"]) * \
                                                            float(conf["NPV"][
                                                                      "OPEX_DISTRIBUTEUR"])
    df_revenues_per_site["opex_clients_commissions_voice"] = (
            df_revenues_per_site["revenue_voice_mobile"] * float(conf["NPV"]["OPEX_DISTRIBUTEUR"]))
    df_revenues_per_site["cost_interco"] = (
            df_revenues_per_site['total_voice_traffic_kerlangs'] *
            (df_revenues_per_site['ratio_prepaid_inwi'] +
             df_revenues_per_site['ratio_postpaid_inwi']) * conf['NPV']['COST_INWI']
            + df_revenues_per_site['total_voice_traffic_kerlangs'] *
            (df_revenues_per_site['ratio_prepaid_iam'] + df_revenues_per_site['ratio_postpaid_iam'])
            * conf['NPV']['COST_IAM'])
    df_revenues_per_site["opex_clients_commissions"] = df_revenues_per_site[
                                                           "opex_clients_commissions_data"] + \
                                                       df_revenues_per_site[
                                                           "opex_clients_commissions_voice"]
    df_revenues_per_site["opex_clients"] = (df_revenues_per_site["opex_clients_commissions"] +
                                            df_revenues_per_site['cost_interco'])
    return df_revenues_per_site


@add_logging_info
def compute_increase_of_arpu_by_the_upgrade(df_final_results_technical_part,
                                            df_unit_prices,
                                            df_post_pre_paid_per_region):
    """
    Compute the increase on ARPU due to the upgrade

    Parameters
    ----------
    revenues_per_unit_traffic: pd.DataFrame
        Dataset with revenue per unit's information
    df_final_results_technical_part: pd.DataFrame
        Result of technical module
    df_unit_prices: pd.DataFrame
        Dataset with in unit price's information

    Returns
    -------
    df_increase_arpu: pd.DataFrame
    """

    df_final_results_technical_part['date'] = pd.to_datetime(
        df_final_results_technical_part['date'], format='%Y-%m-%d')
    df_final_results_technical_part['month'] = df_final_results_technical_part.date.dt.month
    df_final_results_technical_part['month'] = df_final_results_technical_part['month'].apply(
        lambda x: str(x).zfill(2))
    df_final_results_technical_part['year'] = df_final_results_technical_part['year'].apply(str)
    df_final_results_technical_part['month_period'] = df_final_results_technical_part['year'] + \
                                                      df_final_results_technical_part['month']

    df_final_results_technical_part["unit_price_data_box"] = 0
    df_final_results_technical_part["unit_price_data_mobile"] = 0
    df_final_results_technical_part["unit_price_voice_min"] = 0
    years = generate_bp_years(int(conf['OSS_PREPROCESSING']['LAST_YEAR_OF_OSS']))
    for year in years:
        year = str(year)
        unit_price_data_box = \
            df_unit_prices[df_unit_prices["type"] == "arpg_data_box"][year].values[0]
        unit_price_data_mobile = \
            df_unit_prices[df_unit_prices["type"] == "arpg_data_mobile"][year].values[0]
        unit_price_voice_min = \
            df_unit_prices[df_unit_prices["type"] == "arpm_voice_avg"][year].values[0]
        df_final_results_technical_part.loc[df_final_results_technical_part['year'] == year,
        'unit_price_data_box'] = unit_price_data_box
        df_final_results_technical_part.loc[df_final_results_technical_part['year'] == year,
        'unit_price_data_mobile'] = unit_price_data_mobile
        df_final_results_technical_part.loc[df_final_results_technical_part['year'] == year,
        'unit_price_voice_min'] = unit_price_voice_min

    df_final_results_technical_part['unit_price_data_box_with_the_decrease'] = (
        df_final_results_technical_part.unit_price_data_box)
    df_final_results_technical_part['unit_price_data_mobile_with_the_decrease'] = (
        df_final_results_technical_part.unit_price_data_mobile)
    df_final_results_technical_part[
        'unit_price_voice_with_the_decrease'] = df_final_results_technical_part.unit_price_voice_min

    df_final_results_technical_part = (
        pd.merge(df_final_results_technical_part,
                 df_post_pre_paid_per_region[['site_id_x','cluster_key',
                                              'ratio_prepaid','ratio_postpaid']],
                 left_on='site_id', right_on='site_id_x'))


    print(df_final_results_technical_part)
    print(df_post_pre_paid_per_region)


    df_increase_arpu = df_final_results_technical_part.copy()

    # Is unit price per GB or what ?
    ### Transform technical to economical by multiplying the traffic by its unitary price
    df_increase_arpu['arpu_increase_due_to_the_upgrade_data_box'] = \
        df_increase_arpu['traffic_increase_due_to_the_upgrade_data_box'] * \
        df_increase_arpu['unit_price_data_box_with_the_decrease']
    df_increase_arpu['arpu_increase_due_to_the_upgrade_data_mobile'] = \
        df_increase_arpu['traffic_increase_due_to_the_upgrade_data_mobile'] * \
        df_increase_arpu['unit_price_data_mobile_with_the_decrease']

    df_increase_arpu['arpu_increase_due_to_the_upgrade_data'] = \
        df_increase_arpu['arpu_increase_due_to_the_upgrade_data_box'] + \
        df_increase_arpu['arpu_increase_due_to_the_upgrade_data_mobile']

    df_increase_arpu["arpu_increase_due_to_the_upgrade_voice"] = (
            df_increase_arpu["traffic_increase_due_to_the_upgrade_voice"] *
            df_increase_arpu['ratio_prepaid'] * conf['NPV']['ARPM_PREPAID'] +
            df_increase_arpu["traffic_increase_due_to_the_upgrade_voice"] *
            df_increase_arpu['ratio_postpaid'] * conf['NPV']['ARPM_POSTPAID'] * 1000 * 60)

    df_increase_arpu = df_increase_arpu.drop_duplicates(keep="first")

    return df_increase_arpu


def compute_post_pre_paid_per_region(df_cluster_key, df_sites, post_paid_pre_paid_region):
    """
    Compute the post-paid percentage for each region based

    Parameters
    ----------
    df_cluster_key: pd.DataFrame
        Dataset with new site and his neighbour and cluster_key(new column name)
    df_sites: pd.DataFrame
        Site's information
    post_paid_pre_paid_region: pd.DataFrame
        Dataset with information post paid pre paid for each Region

    Returns
    -------
    region_with_postpaid_prepaid: pd.DataFrame
    """
    df_sites = df_sites.drop_duplicates()
    region_with_postpaid_prepaid = pd.merge(df_sites, post_paid_pre_paid_region, on='region')
    region_with_postpaid_prepaid = pd.merge(df_cluster_key, region_with_postpaid_prepaid,
                                            left_on='neighbour_1', right_on='site_id')
    return region_with_postpaid_prepaid
