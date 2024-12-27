import logging

import google.cloud.logging
from train_technical_pipeline import pipeline
from google.cloud import aiplatform
from kfp import compiler
from utils.config import pipeline_config

def train_churn_model():
    """Process the churn training pipeline by compiling and submitting the vertex pipeline job."""

    # google logging client
    client_log = google.cloud.logging.Client()
    client_log.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Save the compiled json file for the pipeline to be used by the vertex pipeline
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_config['package_path'])

    display_name = pipeline_config["display_name"]
    print(f"launching pipeline job {display_name}")

    dict_pipeline_params = {"project_id": pipeline_config["project_id"],
                        "location": pipeline_config["location"], 
                        "m0":pipeline_config["m0"],
                       "dataset_train_table_id": pipeline_config["dataset_train_table_id"],
                       "nb_train_samples_per_class" : pipeline_config["nb_train_samples_per_class"],
                       "nb_test_samples_per_class" : pipeline_config["nb_test_samples_per_class"]}

    job = aiplatform.PipelineJob(project=pipeline_config["project_id"],
                         display_name='churn-prediction-pipeline',
                         enable_caching=True,
                         location=pipeline_config["location"],
                         template_path='churn-prediction-pipeline.json', parameter_values=dict_pipeline_params)
                  
    job.submit(service_account=pipeline_config["pipeline_config"])


if __name__ == "__main__":
    train_churn_model()