import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from preprocess_technical_pipeline import pipeline
from utils.config import pipeline_config


def preprocess_technical_pipeline_request():
    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    parameter_values = {"project_id": pipeline_config["project_id"],
            "location": pipeline_config["location"],
            "processed_sites_table_id": pipeline_config['processed_sites_table_id'],
            "processed_oss_counter_table_id": pipeline_config['processed_oss_counter_table_id'],
            "maximum_weeks_to_group_upgrade": pipeline_config["TRAFFIC_IMPROVEMENT"]["MAXIMUM_WEEKS_TO_GROUP_UPGRADES"],
            "max_number_of_neighbour":  pipeline_config["TRAFFIC_IMPROVEMENT"]["MAX_NUMBER_OF_NEIGHBORS"],
            "cell_affected_table_id":pipeline_config['cell_affected_table_id'],
            "sites_to_remove_table_id":pipeline_config['sites_to_remove_table_id'],
            "list_of_upgrades_table_id":pipeline_config['list_of_upgrades_table_id']}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                         display_name=pipeline_config["display_name"],
                         enable_caching=False,
                         location=pipeline_config["location"],
                         template_path=pipeline_config["package_path"],
                         parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    preprocess_technical_pipeline_request()
