import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from train_technical_pipeline import pipeline
from utils.config import pipeline_config


def train_technical_pipeline_request():
    """Submit the train technical pipeline job to run on Vertex AI."""

    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    dict_parameter_values = {
        "project_id": pipeline_config["project_id"],
        "location": pipeline_config["location"],
        "processed_oss_counter_table_id": pipeline_config['processed_oss_counter_table_id'],
        "data_traffic_features_table_id": pipeline_config['data_traffic_features_table_id'],
        "voice_traffic_features_table_id": pipeline_config['voice_traffic_features_table_id'],
        "data_train_test_predictions_table_id": pipeline_config['data_train_test_predictions_table_id'],
        "voice_train_test_predictions_table_id": pipeline_config['voice_train_test_predictions_table_id'],
        "dict_technical_pipeline": pipeline_config["TRAIN_TECHNICAL_PIPELINE"],
        "exec_time": pipeline_config["TRAIN_TECHNICAL_PIPELINE"]["EXEC_TIME"],
        "models_directory": pipeline_config["models_directory"],
        "kpis_to_compute_trend": pipeline_config["TRAIN_TECHNICAL_PIPELINE"]["KPIS_TO_COMPUTE_TREND"],
        "kpi_to_compute_trend_data": pipeline_config["TRAIN_TECHNICAL_PIPELINE"]["KPI_TO_COMPUTE_TREND_DATA"],
        "kpi_to_compute_trend_voice": pipeline_config["TRAIN_TECHNICAL_PIPELINE"]["KPI_TO_COMPUTE_TREND_VOICE"]}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                         display_name=pipeline_config["display_name"],
                         enable_caching=False,
                         location=pipeline_config["location"],
                         template_path='train-technical-pipeline.json',
                         parameter_values=dict_parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    train_technical_pipeline_request()
