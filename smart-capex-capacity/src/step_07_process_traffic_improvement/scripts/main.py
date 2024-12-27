import functions_framework
import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from config import pipeline_config

@functions_framework.http
def process_traffic_improvement_request(request):
    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    # compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    parameter_values = {
            "project_id": pipeline_config["project_id"],
            "location": pipeline_config["location"],
            "gcs_bucket": pipeline_config['gcs_bucket'],
            "xgboost_data_model_path": pipeline_config['xgboost_data_model_path'],
            "xgboost_voice_model_path": pipeline_config['xgboost_voice_model_path'],
            "trend_data_models_path": pipeline_config['trend_data_models_path'],
            "trend_voice_models_path": pipeline_config['trend_voice_models_path'],
            "sites_table_id": pipeline_config["sites_table_id"],
            "predicted_traffic_kpis_table_id": pipeline_config["predicted_traffic_kpis_table_id"],
            "selected_band_per_site_table_id": pipeline_config["selected_band_per_site_table_id"],
            "data_traffic_improvement_features_table_id": pipeline_config["data_traffic_improvement_features_table_id"],
            "voice_traffic_improvement_features_table_id": pipeline_config["voice_traffic_improvement_features_table_id"],
            "predicted_increase_in_traffic_by_the_upgrade_table_id": pipeline_config["predicted_increase_in_traffic_by_the_upgrade_table_id"],
            "data_kpi_to_compute_effect": pipeline_config["data_kpi_to_compute_effect"],
            "voice_kpi_to_compute_effect": pipeline_config["voice_kpi_to_compute_effect"],
            "variable_to_group_by": pipeline_config["variable_to_group_by"],
            "data_kpi_to_compute_trend": pipeline_config["data_kpi_to_compute_trend"],
            "voice_kpi_to_compute_trend": pipeline_config["voice_kpi_to_compute_trend"],
            "dict_traffic_improvement": pipeline_config['TRAFFIC_IMPROVEMENT'],
            "dict_traffic_improvement_trend": pipeline_config['TRAFFIC_IMPROVEMENT_TREND']}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name=pipeline_config["display_name"],
                                 enable_caching=False,
                                 location=pipeline_config["location"],
                                 template_path=pipeline_config["package_path"],
                                 parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])

    while True:
        is_done = job.state.PIPELINE_STATE_SUCCEEDED == job.state
        if is_done:
            return "1"
