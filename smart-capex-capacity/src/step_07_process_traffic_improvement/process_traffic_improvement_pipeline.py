from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_data import load_data
from components.step_02_prepare_dataset_of_future_upgrades import prepare_dataset_of_future_upgrades
from components.step_03_predict_increase_of_traffic_after_upgrade import predict_increase_of_traffic_after_upgrade
from components.step_04_get_traffic_improvement import get_traffic_improvement
from components.step_05_predict_improvement_traffic_trend_kpis import predict_improvement_traffic_trend_kpis
from components.step_06_merge_predicted_improvement_traffics import merge_predicted_improvement_traffics


@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description=pipeline_config["description"])
def pipeline(project_id: str,
             location: str,
             gcs_bucket: str,
             xgboost_data_model_path: str,
             xgboost_voice_model_path: str,
             trend_data_models_path: str,
             trend_voice_models_path: str,
             sites_table_id: str,
             predicted_traffic_kpis_table_id: str,
             selected_band_per_site_table_id: str,
             data_traffic_improvement_features_table_id: str,
             voice_traffic_improvement_features_table_id: str,
             predicted_increase_in_traffic_by_the_upgrade_table_id: str,
             data_kpi_to_compute_effect: str,
             voice_kpi_to_compute_effect: str,
             variable_to_group_by: str,
             data_kpi_to_compute_trend: str,
             voice_kpi_to_compute_trend: str,
             dict_traffic_improvement: dict,
             dict_traffic_improvement_trend: dict):

    # Load Sites
    load_sites_op = load_data(project_id=project_id,
                              location=location,
                              table_id=sites_table_id).set_display_name("load_selected_band_per_site")

    # Load Selected Band per Site
    load_selected_band_per_site_op = load_data(
        project_id=project_id,
        location=location,
        table_id=selected_band_per_site_table_id).set_display_name("load_selected_band_per_site")
    # Load Predicted Traffic KPIs
    load_predicted_traffic_kpis_op = load_data(
        project_id=project_id,
        location=location,
        table_id=predicted_traffic_kpis_table_id).set_display_name("load_predicted_traffic_kpis")
    # Load Data Traffic Improvement Features
    load_traffic_improvement_features_data_op = load_data(
        project_id=project_id,
        location=location,
        table_id=data_traffic_improvement_features_table_id
        ).set_display_name("load_data_traffic_improvement_features")
    # Load Voice Traffic Improvement Features
    load_traffic_improvement_features_voice_op = load_data(
        project_id=project_id,
        location=location,
        table_id=voice_traffic_improvement_features_table_id
        ).set_display_name("load_voice_traffic_improvement_features")

    # Prepare Dataset of Future Upgrades - Data
    prepare_dataset_of_future_upgrades_data_op = prepare_dataset_of_future_upgrades(
        type_of_traffic="data",
        dict_traffic_improvement=dict_traffic_improvement,
        traffic_features_future_upgrades_data_input=(load_traffic_improvement_features_data_op
                                                     .outputs["query_results_data_output"])
        ).set_display_name("prepare_dataset_of_future_upgrades_data")

    # Prepare Dataset of Future Upgrades - Voice
    prepare_dataset_of_future_upgrades_voice_op = prepare_dataset_of_future_upgrades(
        type_of_traffic= "voice",
        dict_traffic_improvement=dict_traffic_improvement,
        traffic_features_future_upgrades_data_input=(load_traffic_improvement_features_data_op
                                                     .outputs["query_results_data_output"])
        ).set_display_name("prepare_dataset_of_future_upgrades_voice")

    # Predict Increase of Traffic After Upgrade - Data
    predict_increase_of_traffic_after_upgrade_data_op = predict_increase_of_traffic_after_upgrade(
        gcs_bucket=gcs_bucket,
        xgboost_model_path=xgboost_data_model_path,
        type_of_traffic="data",
        x_data_input=prepare_dataset_of_future_upgrades_data_op.outputs["x_data_output"],
        traffic_features_future_upgrades_data_input=(load_traffic_improvement_features_data_op
                                                     .outputs["query_results_data_output"])
        ).set_display_name("predict_increase_of_traffic_after_upgrade_data")
    # Predict Increase of Traffic After Upgrade - Voice
    predict_increase_of_traffic_after_upgrade_voice_op = predict_increase_of_traffic_after_upgrade(
        gcs_bucket=gcs_bucket,
        xgboost_model_path=xgboost_voice_model_path,
        type_of_traffic="voice",
        x_data_input=prepare_dataset_of_future_upgrades_voice_op.outputs["x_data_output"],
        traffic_features_future_upgrades_data_input=(load_traffic_improvement_features_voice_op
                                                     .outputs["query_results_data_output"])
        ).set_display_name("predict_increase_of_traffic_after_upgrade_voice")

    # Get Traffic Improvement - Data
    get_traffic_improvement_data_op = get_traffic_improvement(
        kpi_to_compute_upgrade_effect=data_kpi_to_compute_effect,
        traffic_weekly_kpis_data_input=load_predicted_traffic_kpis_op.outputs["query_results_data_output"],
        selected_band_per_site_data_input=load_selected_band_per_site_op.outputs["query_results_data_output"],
        traffic_features_future_upgrades_data_input=(predict_increase_of_traffic_after_upgrade_data_op
                                                     .outputs["traffic_features_future_upgrades_data_output"])
        ).set_display_name("get_traffic_improvement_data")

    # Get Traffic Improvement - Voice
    get_traffic_improvement_voice_op = get_traffic_improvement(
        kpi_to_compute_upgrade_effect=voice_kpi_to_compute_effect,
        traffic_weekly_kpis_data_input=load_predicted_traffic_kpis_op.outputs["query_results_data_output"],
        selected_band_per_site_data_input=load_selected_band_per_site_op.outputs["query_results_data_output"],
        traffic_features_future_upgrades_data_input=(predict_increase_of_traffic_after_upgrade_voice_op
                                                     .outputs["traffic_features_future_upgrades_data_output"])
        ).set_display_name("get_traffic_improvement_voice")

    # Predict Improvement Traffic Trend KPIs - Data
    predict_improvement_traffic_trend_kpis_data_op = predict_improvement_traffic_trend_kpis(
        variable_to_group_by=variable_to_group_by,
        kpi_to_compute_trend=data_kpi_to_compute_trend,
        gcs_bucket=gcs_bucket,
        models_path=trend_data_models_path,
        dict_traffic_improvement_trend=dict_traffic_improvement_trend,
        sites_data_input=load_sites_op.outputs["query_results_data_output"],
        increase_traffic_after_upgrade_data_input=(
            get_traffic_improvement_data_op.outputs["traffic_weekly_kpis_site_data_output"])
        ).set_display_name("predict_improvement_traffic_trend_kpis_data")

    # Predict Improvement Traffic Trend KPIs - Data
    predict_improvement_traffic_trend_kpis_voice_op = predict_improvement_traffic_trend_kpis(
        variable_to_group_by=variable_to_group_by,
        kpi_to_compute_trend=voice_kpi_to_compute_trend,
        gcs_bucket=gcs_bucket,
        models_path=trend_voice_models_path,
        dict_traffic_improvement_trend=dict_traffic_improvement_trend,
        sites_data_input=load_sites_op.outputs["query_results_data_output"],
        increase_traffic_after_upgrade_data_input=(get_traffic_improvement_voice_op
                                                   .outputs["traffic_weekly_kpis_site_data_output"])
        ).set_display_name("predict_improvement_traffic_trend_kpis_voice")

    # Merge Predicted Improvement Traffics
    merge_predicted_improvement_traffics(
        project_id=project_id,
        location=location,
        predicted_increase_in_traffic_by_the_upgrade_table_id=predicted_increase_in_traffic_by_the_upgrade_table_id,
        data_predicted_kpis_trend_data_input=predict_improvement_traffic_trend_kpis_data_op.outputs["all_data_output"],
        voice_predicted_kpis_trend_data_input=predict_improvement_traffic_trend_kpis_voice_op.outputs["all_data_output"]
        ).set_display_name("merge_predicted_improvement_traffics")
