import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from capacity_technical_to_economical_pipeline import pipeline
from utils.config import pipeline_config

def capacity_technical_to_economical_pipeline_request():
    """
    Compile and submit the capacity technical to economical pipeline to Vertex AI.

    This function sets up Google Cloud logging, compiles the pipeline, and submits it as a job to Vertex AI.
    """
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
        "traffic_weekly_kpis_table_id": pipeline_config["traffic_weekly_kpis_table_id"],
        "unit_price_table_id": pipeline_config["unit_price_table_id"],
        "revenues_per_site_table_id": pipeline_config["revenues_per_site_table_id"],
        "increase_of_arpu_by_the_upgrade_table_id": pipeline_config["increase_of_arpu_by_the_upgrade_table_id"],
        "predicted_increase_in_traffic_by_the_upgrade_table_id":
                                        pipeline_config["predicted_increase_in_traffic_by_the_upgrade_table_id"],
        "last_year_of_oss": pipeline_config["LAST_YEAR_OF_OSS"],
        "opex_comissions_percentage": pipeline_config["OPEX_COMISSIONS_PERCENTAGE"],
        "time_to_compute_npv": pipeline_config["TIME_TO_COMPUTE_NPV"]}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name=pipeline_config["display_name"],
                                 enable_caching=False,
                                 location=pipeline_config["location"],
                                 template_path='capacity-technical-to-economical-pipeline.json',
                                 parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    capacity_technical_to_economical_pipeline_request()
