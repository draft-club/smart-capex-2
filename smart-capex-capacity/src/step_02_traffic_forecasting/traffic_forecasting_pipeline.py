from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_data import load_data
from components.step_02_prepare_train import prepare_train
from components.step_03_fit_and_predict import fit_and_predict
from components.step_04_clip_predicted_traffic_kpis import clip_predicted_traffic_kpis
from components.step_05_post_process_forecasts import post_process_forecasts


# pylint: disable=E1120
@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description="Smart Capex: Traffic Forecasting Pipeline")

# create the pipeline and define parameters
def pipeline(project_id: str,
             location: str,
             processed_oss_counter_table_id: str,
             predicted_traffic_kpis_table_id: str,
             traffic_kpi_3g: list,
             traffic_kpi_4g: list,
             interquartile_coefficient: int,
             cross_validation: bool,
             train_test_split: bool,
             training_ratio: float,
             max_date_to_predict: int,
             exec_time: str,
             country_name: str):
    """It is mainly used to run the pipeline of the components of traffic forecasting

    Args:
        project_id (str):  It holds the project_id on GCP
        location (str): It holds the location assigned to the project on GCP
        processed_oss_counter_table_id (str):  It holds the resource name of processed OSS Counter table on BigQuery
        predicted_traffic_kpis_table_id (str):  It holds the resource name of predicted traffic KPIs table on BigQuery
        traffic_kpi_3g (list): It holds a list of traffic KPIs of 3G
        traffic_kpi_4g (list): It holds a list of traffic KPIs of 4G
        interquartile_coefficient (int): It is a coefficient used to remove the outliers as per the grouped keys
        cross_validation (bool): In case of False: It is used to perform training on the data with performing the forecast 
                                 on the dates shifted by one. In case of True: It is used to perform training on the data
                                 with performing the forecast on the training dates
        train_test_split (bool): It is used to perform training on the data with performing the forecast 
                                 on the test data
        training_ratio (float): It is used to split the data into training and test when cross_validation = True
                                and train_test_split = True
        max_date_to_predict (int): It is used to perform training on the data with performing the forecast on next 52 days
        exec_time (str): It represents the week_date_run
        country_name (str): It is the configured country to get the holidays of the country with prophet
    """

    load_data_op = load_data(project_id=project_id,
                             location=location,
                             processed_oss_counter_table_id=processed_oss_counter_table_id)

    prepare_train_op = prepare_train(traffic_kpi_3g=traffic_kpi_3g,
                                     traffic_kpi_4g=traffic_kpi_4g,
                                     traffic_weekly_kpis_data_input=load_data_op.outputs["traffic_weekly_kpis_data_output"])

    fit_and_predict_op = fit_and_predict(
                                interquartile_coefficient=interquartile_coefficient,
                                cross_validation=cross_validation,
                                train_test_split=train_test_split,
                                training_ratio=training_ratio,
                                max_date_to_predict=max_date_to_predict,
                                country_name=country_name,
                                traffic_weekly_kpis_data_input=prepare_train_op.outputs["traffic_weekly_kpis_data_output"])

    clip_predicted_traffic_kpis_op = clip_predicted_traffic_kpis(
                        predicted_traffic_kpis_data_input=fit_and_predict_op.outputs["predicted_traffic_kpis_data_output"])

    post_process_forecasts(
            project_id=project_id,
            location=location,
            exec_time=exec_time,
            predicted_traffic_kpis_table_id=predicted_traffic_kpis_table_id,
            predicted_traffic_kpis_data_input=clip_predicted_traffic_kpis_op.outputs["predicted_traffic_kpis_data_output"])
