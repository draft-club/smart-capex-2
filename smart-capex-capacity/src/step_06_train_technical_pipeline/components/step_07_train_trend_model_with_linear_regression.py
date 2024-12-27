from kfp.dsl import Dataset, Input, Model, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def train_trend_model_with_linear_regression(variable_to_group_by: list,
                                             kpi_to_compute_trend: str,
                                             model_path: str,
                                             exec_time: str,
                                             traffic_by_region_data_input: Input[Dataset],
                                             model_output: Output[Model]):
    """Train a traffic trend linear regression model for each unique value in the `variable_to_group_by` column
        and save the models to Cloud storage.

    Args:
        variable_to_group_by (list): It holds the list of variables to group by for each separate model.
        kpi_to_compute_trend (str): It holds the KPI to compute the trend.
        model_path (str): It holds the GCS path to save the model.
        exec_time (str): It holds the execution time for versioning the model.
        traffic_by_region_data_input (Input[Dataset]): It holds the input dataset containing traffic by region data.
        model_output (Output[Model]): It holds the output model artifact.
    """

    import os
    import joblib
    import pandas as pd
    from google.cloud import storage
    from sklearn.linear_model import LinearRegression

    def fit_and_save_linear_regression_model(df, kpi_to_compute_trend, variable_to_group_by, model_path, exec_time):
        """Train and save traffic trend linear regression model for a given dataframe group.

        Args:
            df (pd.DataFrame): It holds the DataFrame containing the data.
            kpi_to_compute_trend (str): It holds the KPI to compute the trend.
            variable_to_group_by (list): It holds the list of variables to group by.
            model_path (str): It holds the path to save the model.
            exec_time (str): It holds the execution time for versioning the model.
        """

        df.reset_index(drop=True, inplace=True)
        df["difference_weeks"] = (df["week_date"] - df["week_date"].min()) / pd.Timedelta(1, "W")
        x_train = df[["difference_weeks"]]
        y_train = df[["value_traffic_kpi"]]

        model = LinearRegression()
        model.fit(x_train, y_train)

        model_filename = f"{df[variable_to_group_by[0]][0]}.sav"
        model_path = os.path.join((f"{model_path}/trend_model/traffic_trend_by_{variable_to_group_by[0]}/"
                                  f"{kpi_to_compute_trend}/{exec_time}/{model_filename}"))

        # Save model artifact to local filesystem (doesn't persist)
        local_file_name = model_output.path + ".sav"
        with open(local_file_name, 'wb') as file:
            joblib.dump(model, file)

        blob = storage.blob.Blob.from_string(model_path, client=storage.Client())
        blob.upload_from_filename(local_file_name)

    df_traffic_by_region = pd.read_parquet(traffic_by_region_data_input.path)

    df_traffic_by_region["week_date"] = pd.to_datetime(df_traffic_by_region["week_date"])

    # Transform the dataset to easily apply a groupby afterwards
    df = pd.melt(df_traffic_by_region, id_vars=["week_date"] + variable_to_group_by, value_vars=kpi_to_compute_trend)

    df.columns = ["week_date"] + variable_to_group_by + ["traffic_kpis", "value_traffic_kpi"]

    # Train and save model of the regional trend
    df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(fit_and_save_linear_regression_model,
                                                              kpi_to_compute_trend=kpi_to_compute_trend,
                                                              variable_to_group_by=variable_to_group_by,
                                                              model_path=model_path, exec_time=exec_time)

    # Train and save model of the global trend
    df = df.groupby(["week_date", "traffic_kpis"])["value_traffic_kpi"].sum().reset_index()

    df[variable_to_group_by[0]] = "GLOBAL"

    df.groupby(variable_to_group_by + ["traffic_kpis"]).apply(fit_and_save_linear_regression_model,
                                                              kpi_to_compute_trend=kpi_to_compute_trend,
                                                              variable_to_group_by=variable_to_group_by,
                                                              model_path=model_path, exec_time=exec_time)
