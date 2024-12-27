import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from process_bands_to_upgrade_pipeline import pipeline
from utils.config import pipeline_config


def process_bands_to_upgrade_request():
    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    parameter_values = {
        "project_id": pipeline_config["project_id"],
        "location": pipeline_config["location"],
        "processed_sites_table_id": pipeline_config['processed_sites_table_id'],
        "predicted_traffic_kpis_table_id": pipeline_config['predicted_traffic_kpis_table_id'],
        "b32_table_id": pipeline_config['b32_table_id'],
        "week_of_the_upgrade": pipeline_config["TRAFFIC_IMPROVEMENT"]["WEEK_OF_THE_UPGRADE"],
        "max_number_of_neighbors": pipeline_config["TRAFFIC_IMPROVEMENT"]["MAX_NUMBER_OF_NEIGHBORS"],
        "selected_band_per_site_table_id": pipeline_config['selected_band_per_site_table_id'],
        "affected_cells_table_id": pipeline_config['affected_cells_table_id'],
        "cluster_of_the_upgrade_table_id": pipeline_config['cluster_of_the_upgrade_table_id'],
        "congestion_status_for_db_table_id": pipeline_config["congestion_status_for_db_table_id"]}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name='process-bands-to-upgrade-pipeline.json',
                                 enable_caching=False,
                                 location=pipeline_config["location"],
                                 template_path='process-bands-to-upgrade-pipeline.json',
                                 parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    process_bands_to_upgrade_request()
