from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)

from utils.config import pipeline_config

@component(base_image=pipeline_config["base_image"])
def predict_improvement_traffic_trend_kpis(variable_to_group_by: str,
                                           kpi_to_compute_trend: str,
                                           gcs_bucket: str,
                                           models_path: str,
                                           dict_traffic_improvement_trend: dict,
                                           sites_data_input: Input[Dataset],
                                           increase_traffic_after_upgrade_data_input: Input[Dataset],
                                           all_data_output: Output[Dataset]):

    """Predicts the improvement in traffic trend KPIs after an upgrade.

    Args:
        variable_to_group_by (str): It holds the variable to group the data by.
        kpi_to_compute_trend (str): It holds the KPI to compute the trend.
        gcs_bucket (str): It holds the name of the GCS bucket.
        models_path (str): It holds the path to the model.
        dict_traffic_improvement_trend (dict): It holds the traffic improvement trend parameters.
        sites_data_input (Input[Dataset]): It holds the sites data.
        increase_traffic_after_upgrade_data_input (Input[Dataset]): It holds the traffic data after upgrade.
        all_data_output (Output[Dataset]): It holds the predictions of increase in traffic due to the upgrade.

    Returns:
        all_data_output (Output[Dataset]): It holds the predictions of increase in traffic due to the upgrade.
    """

    import joblib
    import datetime
    import numpy as np
    import pandas as pd
    from google.cloud import storage

    def get_lag_between_two_week_periods(week_period_1, week_period_2):
        week_period_1, week_period_2 = str(int(float(week_period_1))), str(int(float(week_period_2)))
        year1 = int(week_period_1[:4])
        week1 = int(week_period_1[-2:])
        year2 = int(week_period_2[:4])
        week2 = int(week_period_2[-2:])
        return - (53 * year1 + week1 - (53 * year2 + week2))

    def load_model_from_gcs(gcs_bucket, model_path):
        # Load the model pickle file from cloud storage
        client = storage.Client()
        bucket = client.get_bucket(gcs_bucket)
        blob = bucket.blob(model_path)

        with blob.open("rb") as file:
            model = joblib.load(file)

        return model

    def predict_linear_regression(df, gcs_bucket, models_path, kpi_to_compute_trend):

        df = df.reset_index()

        ### HINT: added to adapt to GCS remote model loading
        site_region = df["site_region"].drop_duplicates()[0]
        model_path = f"{models_path}/{site_region}.sav"
        loaded_model = load_model_from_gcs(gcs_bucket, model_path)

        increment = loaded_model.coef_ / loaded_model.intercept_[0]
        increment = increment[0]

        ## Set up a maximum increment per region
        max_weekly_increment = (dict_traffic_improvement_trend["MAX_YEARLY_INCREMENT"] / 100) / 52
        if increment > max_weekly_increment:
            increment = max_weekly_increment

        ## Compute the lag with the week of the upgrade
        df["week_of_the_upgrade"] = df["week_of_the_upgrade"].apply(int).apply(str)
        df["week_period"] = df["week_period"].apply(int).apply(str)
        df["lag_between_upgrade"] = df[["week_of_the_upgrade", "week_period"]] \
            .apply(lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

        ## Create the dataset to put the increase in traffic
        ## Starting date: week of the upgrade
        ## Final date: week of the upgrade + max_weeks_to_predict
        max_date = pd.to_datetime(
            df.loc[df["week_of_the_upgrade"] == df["week_period"]]["week_date"].values[0])
        index = pd.date_range(max_date, periods=dict_traffic_improvement_trend["MAX_WEEKS_TO_PREDICT"], freq="W")
        df_date = pd.DataFrame({"date": index})
        df_date["date"] = df_date["date"] + datetime.timedelta(days=1)

        ### Average of traffic before the upgrade
        df_before = df[(df["lag_between_upgrade"] < 0) & (df["lag_between_upgrade"] >= -8)][kpi_to_compute_trend].mean()
        df_date["traffic_before"] = df_before[0]

        ### Increase of traffic after the upgrade
        df_date["increase_of_traffic_after_the_upgrade"] = df["increase_of_traffic_after_the_upgrade"].drop_duplicates()[0]

        ## The traffic that will grow is equal to the traffic before the upgrade
        # + the increment in traffic due to the upgrade
        df_date["total_traffic_to_compute_increase"] = df_date["traffic_before"] + df_date[
                                                                                  "increase_of_traffic_after_the_upgrade"]
        df_date["increase"] = df_date["increase_of_traffic_after_the_upgrade"] * increment

        df_date["week_of_the_upgrade"] = str(df["week_of_the_upgrade"].drop_duplicates()[0])

        df_date["year"] = df_date.date.dt.strftime("%Y")
        df_date["week"] = df_date.date.dt.strftime("%U")
        df_date["week_period"] = df_date.date.dt.strftime("%Y%U")

        ## Compute the lag between the week of the upgrade and the week
        df_date["lag_between_upgrade"] = df_date[["week_of_the_upgrade", "week_period"]].apply(
            lambda x: get_lag_between_two_week_periods(x.iloc[0], x.iloc[1]), axis=1)

        ## After the maximum number of weeks, the increase will be 0
        df_date["increase"] = np.where(df_date["lag_between_upgrade"] > dict_traffic_improvement_trend[
                                      "MAX_WEEKS_TO_CONSIDER_INCREASE"], 0, df_date["increase"])

        df_date["increase"] = np.where(df_date["lag_between_upgrade"] < dict_traffic_improvement_trend[
                                      "MIN_WEEKS_TO_CONSIDER_INCREASE"], 0, df_date["increase"])

        df_date["total_increase"] = np.cumsum(df_date["increase"])
        df_date["traffic_increase_due_to_the_upgrade"] = (
            df_date["total_increase"] + df_date["total_traffic_to_compute_increase"] - df_date["traffic_before"])


        ## The upgrade effect will take a minimum number of weeks to make effect
        df_date["traffic_increase_due_to_the_upgrade"] = np.where(
            df_date["lag_between_upgrade"] <= dict_traffic_improvement_trend[
                "WEEKS_TO_WAIT_AFTER_UPGRADE"], 0, df_date["traffic_increase_due_to_the_upgrade"])
        return df_date

    kpi_to_compute_trend = [kpi_to_compute_trend]

    df_sites = pd.read_parquet(sites_data_input.path)
    df_increase_traffic_after_upgrade = pd.read_parquet(increase_traffic_after_upgrade_data_input.path)

    # Merge sites with region info
    df_sites.rename(columns={"region": "site_region"}, inplace=True)
    df_sites = (df_sites[["site_id", "site_region", "site_area"]].drop_duplicates().groupby("site_id").first().reset_index())

    df_increase_traffic_after_upgrade = df_increase_traffic_after_upgrade.merge(df_sites, on="site_id", how="left")
    ## If the site region is not available we put the global trend
    df_increase_traffic_after_upgrade[variable_to_group_by].fillna("GLOBAL", inplace=True)

    # For each site predicts the increase in traffic due to the upgrade
    df_all = (df_increase_traffic_after_upgrade
            .groupby(["site_id", "bands_upgraded","site_area"])
            .apply(predict_linear_regression,
            gcs_bucket=gcs_bucket,
            models_path=models_path,
            kpi_to_compute_trend=kpi_to_compute_trend).reset_index())


    df_all.to_parquet(all_data_output.path)
