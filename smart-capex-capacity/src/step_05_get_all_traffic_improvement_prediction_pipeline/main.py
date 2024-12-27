import logging
from kfp import compiler
from google.cloud import aiplatform
import google.cloud.logging

from get_all_traffic_improvement_pipeline import pipeline
from utils.config import pipeline_config


def get_all_traffic_improvement_pipeline_request():
    """_summary_
    """
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
        "traffic_weekly_kpis_table_id": pipeline_config['traffic_weekly_kpis_table_id'],
        "max_number_of_neighbors": pipeline_config["TRAFFIC_IMPROVEMENT"]["MAX_NUMBER_OF_NEIGHBORS"],
        "cell_affected_table_id": pipeline_config['cell_affected_table_id'],
        "sites_to_remove_table_id": pipeline_config['sites_to_remove_table_id'],
        "list_of_upgrades_table_id": pipeline_config['list_of_upgrades_table_id'],
        "bands_to_consider": pipeline_config['TRAFFIC_IMPROVEMENT']['BANDS_TO_CONSIDER'],
        "weeks_to_wait_after_upgrade": pipeline_config["TRAFFIC_IMPROVEMENT"]["WEEKS_TO_WAIT_AFTER_UPGRADE"],
        "weeks_to_wait_after_upgrade_max": pipeline_config["TRAFFIC_IMPROVEMENT"]["WEEKS_TO_WAIT_AFTER_UPGRADE_MAX"],
        "data_traffic_features_table_id": pipeline_config["data_traffic_features_table_id"],
        "voice_traffic_features_table_id": pipeline_config["voice_traffic_features_table_id"]
    }

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name=pipeline_config["display_name"],
                                 enable_caching=False,
                                 location=pipeline_config["location"],
                                 template_path='get-all-traffic-improvement-pipeline.json',
                                 parameter_values=dict_parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    get_all_traffic_improvement_pipeline_request()
