import logging

from google.cloud import aiplatform
import google.cloud.logging

from kfp import compiler
from capacity_economical_pipeline import pipeline
from utils.config import pipeline_config

def process_capacity_economical_pipeline_request():
    """
    Compile and submit the capacity economical pipeline to Vertex AI.

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
        "revenues_per_unit_traffic_table_id": pipeline_config['revenues_per_unit_traffic_table_id'],
        "increase_arpu_due_to_the_upgrade_table_id": pipeline_config['increase_arpu_due_to_the_upgrade_table_id'],
        "processed_opex_table_id": pipeline_config['processed_opex_table_id'],
        "processed_capex_table_id": pipeline_config["processed_capex_table_id"],
        "margin_per_site_table_id": pipeline_config["margin_per_site_table_id"],
        "increase_arpu_by_year_table_id": pipeline_config['increase_arpu_by_year_table_id'],
        "cash_flow_table_id": pipeline_config['cash_flow_table_id'],
        "npv_of_the_upgrade_table_id": pipeline_config['npv_of_the_upgrade_table_id'],
        "time_to_compute_npv": pipeline_config["NPV"]['TIME_TO_COMPUTE_NPV'],
        "wacc": pipeline_config["NPV"]['WACC'],
}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                                 display_name=pipeline_config["display_name"],
                                 enable_caching=True,
                                 location=pipeline_config["location"],
                                 template_path=pipeline_config["package_path"],
                                 parameter_values=parameter_values)

    job.submit(pipeline_config["service_account"])


if __name__ == "__main__":
    process_capacity_economical_pipeline_request()
