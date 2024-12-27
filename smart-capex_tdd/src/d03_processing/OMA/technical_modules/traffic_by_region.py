import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.d00_conf.conf import conf
from src.d01_utils.utils import getlastfile, check_path, add_logging_info





def compute_traffic_by_region(
        df_traffic_weekly_kpis,
        kpi_to_compute_trend="total_data_traffic_dl_gb"):
    """
    Function to compute traffic by region

    Parameters
    ----------
    df_traffic_weekly_kpis: pd.DataFrame
        OSS weekly data
    kpi_to_compute_trend:  pd.DataFrame
        variable present in the df_traffic_weekly_kpis that you want to group

    Returns
    ----------
    df: pd.DataFrame
        OSS weekly data groupby region and date

    """
    # TMP TO adapt les range de date
    df_traffic_weekly_kpis['date'] = df_traffic_weekly_kpis['date'].astype(str)
    df_traffic_weekly_kpis['date'] = df_traffic_weekly_kpis['date'].str.split('/').str[0]

    # Add date and rename regions
    df_traffic_weekly_kpis["date"] = pd.to_datetime(df_traffic_weekly_kpis["date"])
    df_traffic_weekly_kpis.rename(columns={"region": "site_region"}, inplace=True)

    ## Groupby site and date and sum the kpi to compute the tred
    df = df_traffic_weekly_kpis.groupby(
        ["site_region", "date"])[kpi_to_compute_trend].sum().reset_index()
    df_list_date = list(df.date.unique())[:-1]
    df = df[df.date.isin(df_list_date)]
    return df


def train_trend_model_with_linear_regression(
    df_traffic_by_region,
    output_route='fake',
    variable_to_group_by="site_region",
    kpi_to_compute_trend="total_data_traffic_dl_gb"):

    """
    The train_trend_model_with_linear_regression function trains a linear regression model to
    compute traffic trends for different regions and globally. It processes the input data,
    groups it by specified variables, and applies the linear_regression function to train and save
    the models

    Parameters
    ----------
    df_traffic_by_region: pd.Dataframe
        dataset with the traffic by region and date
    output_route: str
        route to save the model
    variable_to_group_by: list
        the variable to group the OSS data
    kpi_to_compute_trend: list
        the KPI used to compute trend

    """
    # Get Variable
    variable_to_group_by =[variable_to_group_by]
    kpi_to_compute_trend = [kpi_to_compute_trend ]
    df_traffic_by_region["date"] = pd.to_datetime(df_traffic_by_region["date"])

    # Transform the dataset to easily apply a groupby afterwards
    df = pd.melt(
        df_traffic_by_region,
        id_vars=["date"] + variable_to_group_by,
        value_vars=kpi_to_compute_trend,
    )

    # define columns
    df.columns = ["date"] + variable_to_group_by + ["traffic_kpis", "value_traffic_kpi"]
    df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(
        linear_regression,
        variable_to_group_by=variable_to_group_by,
        output_route=output_route,
        kpi_to_compute_trend=kpi_to_compute_trend
    )

    # Train the global trend
    df = df.groupby(["date", "traffic_kpis"])["value_traffic_kpi"].sum().reset_index()
    df[variable_to_group_by[0]] = "GLOBAL"
    df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(
        linear_regression,
        variable_to_group_by=variable_to_group_by,
        output_route=output_route,
        kpi_to_compute_trend=kpi_to_compute_trend
    )


def linear_regression(df, variable_to_group_by, output_route, kpi_to_compute_trend):
    """
    This function train linear model to train the global trend

    Parameters
    ----------
    df: pd.DataFrame
    variable_to_group_by: list
    output_route: str
    kpi_to_compute_trend: list

    Returns
    -------

    """
    df = df.reset_index(drop=True)
    min_date = df["date"].min()
    df["difference_weeks"] = (df["date"] - min_date) / pd.Timedelta(1, "W")
    x_train = df[["difference_weeks"]]
    y_train = df[["value_traffic_kpi"]]
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(
        df[variable_to_group_by[0]].drop_duplicates()[0],
        " | coef : ",
        model.coef_,
        " | intercept : ",
        model.intercept_,
        " | R² : ",
        model.score(x_train, y_train),
    )
    filename = df[variable_to_group_by[0]].drop_duplicates()[0] + ".joblib"
    filename_df = df[variable_to_group_by[0]].drop_duplicates()[0] + ".csv"
    trend_model_path = os.path.join(
        output_route,
        "traffic_trend_by_" + variable_to_group_by[0],
        kpi_to_compute_trend[0],
        conf["EXEC_TIME"],
    )
    if not os.path.exists(trend_model_path):
        print(trend_model_path)
        os.makedirs(trend_model_path)
    joblib.dump(model, os.path.join(trend_model_path, filename))
    df.to_csv(os.path.join(trend_model_path, filename_df), index=False, sep='|')

@add_logging_info
def train_regional_model(df_traffic_weekly_kpis):
    """
    The train_regional_model function processes weekly traffic data, groups it by region,
    and trains linear regression models to compute traffic trends for both data and voice traffic.

    Parameters
    ----------
    df_traffic_weekly_kpis: pd.DataFrame
        A pd.DataFrame containing OSS weekly data with columns for date, region, and traffic KPIs.

    Returns
    -------
    Trained linear regression models for both data and voice traffic trends,
    saved to the specified output path.
    """
    # Function to compute traffic by region
    df_traffic_weekly_kpis_groupby = compute_traffic_by_region(df_traffic_weekly_kpis,
                                                               kpi_to_compute_trend=[
                                                                   "total_data_traffic_dl_gb"])
    # For each region train a linear regression model to compute the trend
    train_trend_model_with_linear_regression(df_traffic_weekly_kpis_groupby,
                                             output_route=conf["PATH"]["MODELS"],
                                             variable_to_group_by="site_region",
                                             kpi_to_compute_trend="total_data_traffic_dl_gb")

    df_traffic_weekly_kpis_groupby_voice = compute_traffic_by_region(df_traffic_weekly_kpis,
                                                               kpi_to_compute_trend=[
                                                                   "total_voice_traffic_kerlangs"])
    # For each region train a linear regression model to compute the trend
    train_trend_model_with_linear_regression(df_traffic_weekly_kpis_groupby_voice,
                                             output_route=conf["PATH"]["MODELS"],
                                             variable_to_group_by="site_region",
                                             kpi_to_compute_trend="total_voice_traffic_kerlangs")

def predict_linear_regression(df, parameters):
    """
    The predict_linear_regression function predicts the increase in traffic for a given site after
    an upgrade using a pre-trained linear regression model. It calculates the traffic increment
    based on the model's coefficients and various parameters,
    and adjusts the increment based on specified constraints.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing traffic data and upgrade information.
    parameters : dict
        Dictionary containing model path and various constraints for the prediction.
    Returns
    -------
    df_date: pd.DataFrame
        DataFrame with columns for traffic before the upgrade, increase in traffic,
        and total traffic after the upgrade.
    """
    df = df.reset_index()
    model_path = parameters.get("model_path")
    max_yearly_increment = parameters.get("max_yearly_increment")
    max_weeks_to_consider_increase = parameters.get("max_weeks_to_consider_increase")
    min_weeks_to_consider_increase = parameters.get("min_weeks_to_consider_increase")
    weeks_to_wait_after_the_upgrade = parameters.get("weeks_to_wait_after_the_upgrade")
    kpi_to_compute_trend = parameters.get("kpi_to_compute_trend")[0]
    loaded_model = joblib.load(os.path.join(
        model_path, df["site_region"].drop_duplicates()[0] + ".joblib"
    ))
    increment = loaded_model.coef_ / loaded_model.intercept_[0]
    increment = increment[0]
    ## Set up a maximum increment per region
    max_weekly_increment = (max_yearly_increment / 100) / 52
    increment = min(increment, max_weekly_increment)
    index = df['date']
    df_date = pd.DataFrame({"date": index})
    ### Average of traffic before the upgrade
    if kpi_to_compute_trend == "total_voice_traffic_kerlangs":
        df_date["traffic_before"] = df.n_voice_before.drop_duplicates()[0]
        df_date["increase_of_traffic_after_the_upgrade"] = df[
            "trafic_improvement_voice"
        ].drop_duplicates()[0]
    else:
        df_date["traffic_before"] = df.n_traffic_before.drop_duplicates()[0]
        df_date["increase_of_traffic_after_the_upgrade"] = df[
            "trafic_improvement"
        ].drop_duplicates()[0]
    ## The traffic that will grow is equal to the traffic before the upgrade +
    # the increment in traffic due to the upgrade
    df_date["total_traffic_to_compute_increase"] = (
            df_date["traffic_before"] + df_date["increase_of_traffic_after_the_upgrade"]
    )
    df_date["increase"] = (
            df_date["increase_of_traffic_after_the_upgrade"] * increment
    )
    df_date["week_of_the_upgrade"] = str(
        df["week_of_the_upgrade"].drop_duplicates()[0]
    )
    ## Compute date features
    df_date["year"] = df_date.date.dt.strftime("%Y")
    df_date["week"] = df_date.date.dt.strftime("%U")
    df_date["week_period"] = df_date.date.dt.strftime("%Y%U")
    ## Compute the lag between the week of the upgrade and the week
    df_date["lag_between_upgrade"] = df.lag_between_the_upgrade
    ## After the maximum number of weeks, the increase will be 0
    df_date["increase"] = np.where(
        df_date["lag_between_upgrade"] > max_weeks_to_consider_increase,
        0,
        df_date["increase"],
    )
    df_date["increase"] = np.where(
        df_date["lag_between_upgrade"] < min_weeks_to_consider_increase,
        0,
        df_date["increase"],
    )
    df_date["total_increase"] = np.cumsum(df_date["increase"])
    df_date["traffic_increase_due_to_the_upgrade"] = (
            df_date["total_increase"]
            + df_date["total_traffic_to_compute_increase"]
            - df_date["traffic_before"]
    )
    ## The upgrade effect will take a minimum number of weeks to make effect
    df_date["traffic_increase_due_to_the_upgrade"] = np.where(
        df_date["lag_between_upgrade"] <= weeks_to_wait_after_the_upgrade,
        0,
        df_date["traffic_increase_due_to_the_upgrade"],
    )
    return df_date


@add_logging_info
def predict_improvement_traffic_trend_kpis(
        df_increase_traffic_after_upgrade,
        df_sites,
        params):
    """
    The predict_improvement_traffic_trend_kpis function predicts the increase in traffic for various
    sites after an upgrade using a pre-trained linear regression model. It merges site information
    with traffic data, applies the prediction model,
    and saves the results to a specified output directory.

    Parameters
    ----------
    df_increase_traffic_after_upgrade: pd.DataFrame
        DataFrame containing traffic increase data after upgrades.
    df_sites: pd.DataFrame
        DataFrame containing site information.
    params: dict
        Dictionary containing various parameters like model path, output route,
        and constraints for prediction.

    Returns
    -------
    df_all: pd.DataFrame
        DataFrame containing the predicted increase in traffic for each site after the upgrade.
    """
    model_path = params['model_path']
    output_route = params['output_route']
    max_yearly_increment = params['max_yearly_increment']
    max_weeks_to_consider_increase = params['max_weeks_to_consider_increase']
    min_weeks_to_consider_increase = params['min_weeks_to_consider_increase']
    weeks_to_wait_after_the_upgrade = params['weeks_to_wait_after_the_upgrade']
    variable_to_group_by = params.get('variable_to_group_by', "site_region")
    kpi_to_compute_trend = params.get('kpi_to_compute_trend', "total_data_traffic_dl_gb")
    variable_to_group_by = [variable_to_group_by]
    kpi_to_compute_trend = [kpi_to_compute_trend]
    ## Merge sites with region info
    df_sites.rename(columns={"region": "site_region"}, inplace=True)
    df_sites = (
        df_sites[["site_id", "site_region"]]
        .drop_duplicates()
        .groupby("site_id")
        .first()
        .reset_index()
    )
    df_increase_traffic_after_upgrade = df_increase_traffic_after_upgrade.merge(
        df_sites, on="site_id", how="left"
    )
    ## If the site region is not available we put the global trend
    df_increase_traffic_after_upgrade[variable_to_group_by[0]].fillna(
        "GLOBAL", inplace=True
    )
    path = os.path.join(
        model_path,
        "traffic_trend_by_" + variable_to_group_by[0],
        kpi_to_compute_trend[0],
    )
    path = os.path.join(path, getlastfile(path))



    ## For each site predicts the increase in traffic due to the upgrade
    parameters = {
        'model_path':path,
        'max_yearly_increment': max_yearly_increment,
        'max_weeks_to_consider_increase': max_weeks_to_consider_increase,
        'min_weeks_to_consider_increase': min_weeks_to_consider_increase,
        'weeks_to_wait_after_the_upgrade': weeks_to_wait_after_the_upgrade,
        'kpi_to_compute_trend': kpi_to_compute_trend
    }
    df_all = (
        df_increase_traffic_after_upgrade.groupby(["site_id", "bands_upgraded"])
        .apply(
            predict_linear_regression,
            parameters
        )
        .reset_index()
    )

    ## Save the results in the output route
    final_increase_traffic = os.path.join(
        output_route, "increase_in_traffic_due_to_the_upgrade", conf["EXEC_TIME"]
    )

    if not os.path.exists(final_increase_traffic):
        os.makedirs(final_increase_traffic)

    if kpi_to_compute_trend[0] == 'total_voice_traffic_kerlangs':
        df_all.to_csv(
            os.path.join(final_increase_traffic,
                         "increase_in_traffic_voice_due_to_the_upgrade.csv"),
            sep="|",
            index=False)
    else:
        df_all.to_csv(
        os.path.join(final_increase_traffic, "increase_in_traffic_due_to_the_upgrade.csv"), sep="|",
        index=False)

    return df_all


@add_logging_info
def split_data_traffic(df_predicted_increase_in_traffic_by_the_upgrade):
    """
    Split the increase between the mobile and the box data traffic

    Parameters
    -----------
    df_predicted_increase_in_traffic_by_the_upgrade: pd.DataFrame
        the traffic improvment file at weekly granularity

    Returns
    -------
    df_predicted_increase_in_traffic_by_the_upgrade_merge: pd.DataFrame
        the traffic improvment file at weekly granularity with a split data box/mobile
    """
    # Read the mobile/box weight file
    df_weight = pd.read_csv(
        os.path.join(conf["PATH"]["INTERMEDIATE_DATA"], "df_traffic_box_mobile_year.csv"), sep=";",
        decimal=",")
    df_weight = df_weight[['year', 'weight_box', 'weight_mobile']]
    df_weight.year = df_weight.year.astype(str)
    # Bring weight to the file
    df_predicted_increase_in_traffic_by_the_upgrade_merge =\
        df_predicted_increase_in_traffic_by_the_upgrade.merge(
        df_weight, left_on=['year'], right_on=['year'], how="left")
    # Apply the weight
    df_predicted_increase_in_traffic_by_the_upgrade_merge[
        'traffic_increase_due_to_the_upgrade_data_mobile'] = (
        df_predicted_increase_in_traffic_by_the_upgrade_merge.traffic_increase_due_to_the_upgrade *
        df_predicted_increase_in_traffic_by_the_upgrade_merge.weight_mobile)
    df_predicted_increase_in_traffic_by_the_upgrade_merge[
        'traffic_increase_due_to_the_upgrade_data_box'] = (
        df_predicted_increase_in_traffic_by_the_upgrade_merge.traffic_increase_due_to_the_upgrade *
        df_predicted_increase_in_traffic_by_the_upgrade_merge.weight_box)
    # Save generated file
    check_path(os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                            "increase_in_traffic_due_to_the_upgrade_splitted", conf["EXEC_TIME"]))
    df_predicted_increase_in_traffic_by_the_upgrade_merge.to_csv(
        os.path.join(conf["PATH"]["MODELS_OUTPUT"],
                     "increase_in_traffic_due_to_the_upgrade_splitted", conf["EXEC_TIME"],
                     "df_predicted_increase_in_traffic_due_to_the_upgrade_spllited_" + conf[
                         "USE_CASE"] + "_from_capacity.csv"), sep="|", index=False)
    return df_predicted_increase_in_traffic_by_the_upgrade_merge


def apply_rate_for_new_site(rate, df, cluster_key, site):
    """
    The apply_rate_for_new_site function calculates the traffic improvement for new sites by merging
    data from multiple DataFrames, applying rates based on regions and years, and returning the
    updated DataFrame with the calculated improvements.

    Parameters
    ----------
    rate: pd.DataFrame
        DataFrame containing rate values for different regions and years.
    df : pd.DataFrame
        DataFrame containing site data with traffic improvement values.
    cluster_key : pd.DataFrame
        DataFrame mapping site IDs to their neighboring site IDs.
    site : pd.DataFrame
        DataFrame containing site IDs and their corresponding regions

    Returns
    -------
    result : pd.DataFrame
        Returns a DataFrame with the original data and additional columns for rate and traffic
        improvement with rate.

    """
    add_region = pd.merge(df, cluster_key[['site_id','neighbour_1']], on='site_id')
    add_region = pd.merge(add_region, site[['site_id','region']],
                          left_on='neighbour_1', right_on='site_id')
    add_region['date_date'] = pd.to_datetime(add_region['date'])
    add_region['year'] = add_region['date_date'].dt.year
    add_region_groupby = add_region.groupby(['site_id_x','year']).agg({'trafic_improvement':'sum',
                                                                   'region': 'first'})
    years = [2024, 2025, 2026, 2027, 2028]
    rate['year'] = years[:len(rate)]
    rate_long_df = rate.melt(id_vars='year', var_name='region', value_name='rate')
    merged_rate = add_region_groupby.reset_index().merge(rate_long_df, on=['region', 'year'])
    merged_rate.rename(columns={'site_id_x': 'site_id'}, inplace=True)
    merged_rate['trafic_improvement_with_rate'] = (
            merged_rate['rate'] * merged_rate['trafic_improvement'])
    df['date_date'] = pd.to_datetime(df['date'])
    df['year'] = df['date_date'].dt.year
    result = df.merge(merged_rate[['site_id','rate','year','trafic_improvement_with_rate']],
                      on=['site_id','year'])
    return result

def apply_rate_for_new_site_voice(rate, df, cluster_key, site):
    """
    The apply_rate_for_new_site_voice function calculates the traffic improvement for new site
    voices by merging dataframes, grouping data, and applying rates based on regions and years.

    Parameters
    ----------
    rate: pd.DataFrame
        DataFrame containing rate values for different regions and years.
    df : pd.DataFrame
        DataFrame containing site data with traffic improvement values.
    cluster_key : pd.DataFrame
        DataFrame mapping site IDs to their neighboring site IDs.
    site : pd.DataFrame
        DataFrame containing site IDs and their corresponding regions

    Returns
    -------
    result : pd.DataFrame
        Returns a DataFrame with the original data and additional columns for rate and traffic
        improvement with rate.
    """
    add_region = pd.merge(df, cluster_key[['site_id','neighbour_1']], on='site_id')
    add_region = pd.merge(add_region, site[['site_id','region']], left_on='neighbour_1',
                          right_on='site_id')
    add_region['date_date'] = pd.to_datetime(add_region['date'])
    add_region['year'] = add_region['date_date'].dt.year
    add_region_groupby = (add_region.groupby(['site_id_x','year']).
                          agg({'trafic_improvement_voice':'sum',
                               'region': 'first'}))
    years = [2024, 2025, 2026, 2027, 2028]
    rate['year'] = years[:len(rate)]
    rate_long_df = rate.melt(id_vars='year', var_name='region', value_name='rate')
    merged_rate = add_region_groupby.reset_index().merge(rate_long_df, on=['region', 'year'])
    merged_rate.rename(columns={'site_id_x': 'site_id'}, inplace=True)
    merged_rate['trafic_improvement_with_rate'] = (merged_rate['rate'] *
                                                   merged_rate['trafic_improvement_voice'])
    df['date_date'] = pd.to_datetime(df['date'])
    df['year'] = df['date_date'].dt.year
    result = df.merge(merged_rate[['site_id','rate','year','trafic_improvement_with_rate']],
                      on=['site_id','year'])
    return result
