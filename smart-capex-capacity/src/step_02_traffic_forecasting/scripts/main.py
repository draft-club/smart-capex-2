import functions_framework
import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler

# from traffic_forecasting_pipeline import pipeline
from config import pipeline_config

@functions_framework.http
def process_traffic_forecasting_request(request):
    """It is used to submit the traffic forecasting pipeline
    """
    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    # compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    parameter_values = {"project_id": pipeline_config["project_id"],
                        "location": pipeline_config["location"],
                        "processed_oss_counter_table_id": pipeline_config["processed_oss_counter_table_id"],
                        "predicted_traffic_kpis_table_id": pipeline_config["predicted_traffic_kpis_table_id"],
                        "traffic_kpi_3g": pipeline_config["TRAFFIC_KPI"]["TRAFFIC_3G"],
                        "traffic_kpi_4g": pipeline_config["TRAFFIC_KPI"]["TRAFFIC_4G"],
                        "interquartile_coefficient": pipeline_config["TRAFFIC_FORECASTING"]["INTERQUARTILE_COEFF"],
                        "cross_validation": pipeline_config["TRAFFIC_FORECASTING"]["_CROSS_VALIDATION"],
                        "train_test_split": pipeline_config["TRAFFIC_FORECASTING"]["_TRAIN_TEST_SPLIT"],
                        "training_ratio": pipeline_config["TRAFFIC_FORECASTING"]["_TRAINING_RATIO"],
                        "max_date_to_predict": pipeline_config["TRAFFIC_FORECASTING"]["MAX_DATE_TO_PREDICT"],
                        "exec_time": pipeline_config['TRAFFIC_FORECASTING']['EXEC_TIME'],
                        "country_name": pipeline_config['TRAFFIC_FORECASTING']['COUNTRY_NAME']
                        }

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                         display_name=pipeline_config["display_name"],
                         enable_caching=False,
                         location=pipeline_config["location"],
                         template_path=pipeline_config['package_path'],
                         parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])

    # while True:
    #     is_done = job.state.PIPELINE_STATE_SUCCEEDED == job.state
    #     if is_done:
    #         return "1"

    return ''