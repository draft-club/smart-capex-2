import functions_framework
import logging

import google.cloud.logging
from google.cloud import aiplatform
from kfp import compiler
from config import pipeline_config

@functions_framework.http
def process_data_preparation_request(request):
    """Process the data preparation request by compiling and submitting the vertex pipeline job."""

    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    # compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    dict_parameter_values = {
        "project_id": pipeline_config["project_id"],
        "location": pipeline_config["location"],
        "deployment_history_table_id": pipeline_config["deployment_history_table_id"],
        "oss_3g_table_id": pipeline_config["oss_3G_table_id"],
        "oss_3g_history_table_id": pipeline_config["oss_3G_history_table_id"],
        "oss_4g_table_id": pipeline_config["oss_4G_table_id"],
        "oss_4g_history_table_id": pipeline_config["oss_4G_history_table_id"],
        "processed_sites_table_id": pipeline_config["processed_sites_table_id"],
        "processed_oss_counter_table_id": pipeline_config["processed_oss_counter_table_id"],
        "cells_not_to_consider_table_id": pipeline_config["cells_not_to_consider_table_id"],
        "recent_weeks_threshold": pipeline_config["OSS_PREPROCESSING"]["NB_WEEKS_RECENT"],
        "unmounted_weeks_threshold": pipeline_config["OSS_PREPROCESSING"]["NB_WEEKS_UNMOUNTED"],
        "missing_weeks_threshold": pipeline_config["OSS_PREPROCESSING"]["NB_MISSING_WEEKS_ALLOWED"],
        "variation_coefficient_threshold": pipeline_config["OSS_PREPROCESSING"]["VARIATION_COEFFICIENT_THRESHOLD"],
        "capex_table_id": pipeline_config["capex_table_id"],
        "opex_table_id": pipeline_config["opex_table_id"],
        "processed_capex_table_id": pipeline_config["processed_capex_table_id"],
        "processed_opex_table_id": pipeline_config["processed_opex_table_id"]
        }

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name=pipeline_config["display_name"],
                                 enable_caching=False,
                                 location=pipeline_config["location"],
                                 template_path=pipeline_config["package_path"],
                                 parameter_values=dict_parameter_values)

    job.submit(pipeline_config["service_account"])

    while True:
        is_done = job.state.PIPELINE_STATE_SUCCEEDED == job.state
        if is_done:
            return "1"


