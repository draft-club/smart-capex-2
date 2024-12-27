from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_data import load_data
from components.step_02_prepare_dataset_of_past_upgrades import prepare_dataset_of_past_upgrades
from components.step_03_train_traffic_improvement_model import train_traffic_improvement_model
from components.step_04_get_model import get_model
from components.step_05_evaluate_model import evaluate_model
from components.step_06_compute_traffic_by_region import compute_traffic_by_region
from components.step_07_train_trend_model_with_linear_regression import train_trend_model_with_linear_regression


@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description=pipeline_config["description"])

def pipeline(project_id: str,
             location: str,
             processed_oss_counter_table_id: str,
             data_traffic_features_table_id: str,
             voice_traffic_features_table_id:str,
             data_train_test_predictions_table_id:str,
             voice_train_test_predictions_table_id:str,
             dict_technical_pipeline: dict,
             exec_time: str,
             models_directory: str,
             kpis_to_compute_trend: list,
             kpi_to_compute_trend_data: str,
             kpi_to_compute_trend_voice: str):
    """Define a vertex pipeline for training and evaluating traffic improvement models.

    Args:
        project_id (str): GCP project ID.
        location (str): GCP location.
        processed_oss_counter_table_id (str): Table ID for processed OSS counter data.
        data_traffic_features_table_id (str): Table ID for data traffic features.
        voice_traffic_features_table_id (str): Table ID for voice traffic features.
        data_train_test_predictions_table_id (str): Table ID for data train-test predictions.
        voice_train_test_predictions_table_id (str): Table ID for voice train-test predictions.
        dict_technical_pipeline (dict): Dictionary containing technical pipeline configurations.
        exec_time (str): Execution time for versioning.
        models_directory (str): Directory to save models.
        kpis_to_compute_trend (list): List of KPIs to compute the trend.
        kpi_to_compute_trend_data (str): KPI to compute the trend for data.
        kpi_to_compute_trend_voice (str): KPI to compute the trend for voice.
    """

    # Load Processed OSS Counter
    load_data_oss_counter_op = load_data(project_id=project_id,
                                         location=location,
                                         table_id=processed_oss_counter_table_id
                                         ).set_display_name("load_oss_counter_data")


    # Load Data Traffic Features
    load_data_traffic_features_op = load_data(project_id=project_id,
                                              location=location,
                                              table_id=data_traffic_features_table_id
                                              ).set_display_name("load_data_features")


    # Load Voice Traffic Features
    load_voice_traffic_features_op = load_data(project_id=project_id,
                                               location=location,
                                               table_id=voice_traffic_features_table_id
                                               ).set_display_name("load_voice_features")


    # Prepare Dataset - Data
    prepare_dataset_data_op = prepare_dataset_of_past_upgrades(
                            type_of_traffic="data",
                            remove_samples_with_target_variable_lower=True,
                            dict_technical_pipeline=dict_technical_pipeline,
                            traffic_features_data_input=load_data_traffic_features_op.outputs["query_results_data_output"]
                            ).set_display_name("preprocess_data")


    # Prepare Dataset - Voice
    prepare_dataset_voice_op = prepare_dataset_of_past_upgrades(
                            type_of_traffic="voice",
                            remove_samples_with_target_variable_lower=True,
                            dict_technical_pipeline=dict_technical_pipeline,
                            traffic_features_data_input=load_voice_traffic_features_op.outputs["query_results_data_output"]
                            ).set_display_name("preprocess_voice")


    # Train Model - Data
    train_traffic_improvement_model(type_of_traffic="data",
                                    model_path=models_directory,
                                    exec_time=exec_time,
                                    x_data_input=prepare_dataset_data_op.outputs["x_data_output"],
                                    y_data_input=prepare_dataset_data_op.outputs["y_data_output"]
                                    ).set_display_name("train_data_model")


    # Get prediction - Data
    get_data_model_data_op = get_model(project_id=project_id,
                                       location=location,
                                       train_test_predictions_table_id=data_train_test_predictions_table_id,
                                       x_data_input=prepare_dataset_data_op.outputs["x_data_output"],
                                       y_data_input=prepare_dataset_data_op.outputs["y_data_output"]
                                       ).set_display_name("get_predictions_data_model")


    # Evaluate Model - Data
    evaluate_model(train_test_predictions_data_input=get_data_model_data_op.outputs["train_test_predictions_data_output"],
                   model_input= get_data_model_data_op.outputs["model_output"]).set_display_name("evaluate_voice_model")


    # Train Model - Voice
    train_traffic_improvement_model(type_of_traffic="voice",
                                    model_path=models_directory,
                                    exec_time=exec_time,
                                    x_data_input=prepare_dataset_voice_op.outputs["x_data_output"],
                                    y_data_input=prepare_dataset_voice_op.outputs["y_data_output"]
                                    ).set_display_name("train_voice_model")


    # Get prediction - Voice
    get_voice_model_data_op = get_model(project_id=project_id,
                                        location=location,
                                        train_test_predictions_table_id=voice_train_test_predictions_table_id,
                                        x_data_input=prepare_dataset_voice_op.outputs["x_data_output"],
                                        y_data_input=prepare_dataset_voice_op.outputs["y_data_output"]
                                        ).set_display_name("get_predictions_voice_model")


    # Evaluate Model - Voice
    evaluate_model(train_test_predictions_data_input=get_voice_model_data_op.outputs["train_test_predictions_data_output"],
                   model_input=get_voice_model_data_op.outputs["model_output"]).set_display_name("evaluate_voice_model")


    # Comupte Traffic by Region
    compute_traffic_by_region_op = compute_traffic_by_region(
                                kpis_to_compute_trend=kpis_to_compute_trend,
                                traffic_weekly_kpis_data_input=load_data_oss_counter_op.outputs["query_results_data_output"]
                                ).set_display_name("compute_traffic_by_region")


    train_trend_model_with_linear_regression(
                            variable_to_group_by=["site_region"],
                            kpi_to_compute_trend=kpi_to_compute_trend_data,
                            model_path=models_directory,
                            exec_time=exec_time,
                            traffic_by_region_data_input=compute_traffic_by_region_op.outputs["region_traffic_data_output"]
                            ).set_display_name("train_trend_data_model")


    train_trend_model_with_linear_regression(
                            variable_to_group_by=["site_region"],
                            kpi_to_compute_trend=kpi_to_compute_trend_voice,
                            model_path=models_directory,
                            exec_time=exec_time,
                            traffic_by_region_data_input=compute_traffic_by_region_op.outputs["region_traffic_data_output"]
                            ).set_display_name("train_trend_voice_model")
