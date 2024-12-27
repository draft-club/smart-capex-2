from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def get_capacity_kpis_data(project_id:str, location:str, m0: int, delta: int, congestion_kpis_output: Output[Dataset]
            ):
    
    import pandas as pd
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id, location=location)
    m = (m0 - delta)%12
    query_congestion_kpis = """SELECT site_id, AVG(average_number_of_users_in_queue) as average_nb_user_in_queue, AVG(average_throughput_user_dl) as average_throughput_user_dl, MAX(average_number_of_users_in_queue) as max_nb_user_in_queue, MIN(average_throughput_user_dl) as min_throughput_user_dl FROM `oro-smart-capex-001-dev.intermediate_results.data_preparation_oss_counter_agg_t` where year =2024 and month=""" + str(m) + """ group by site_id"""
      
    congestion_kpis = client.query(query_congestion_kpis).to_dataframe()
    congestion_kpis = congestion_kpis.drop_duplicates()
    print(len(congestion_kpis))
    congestion_kpis = congestion_kpis.dropna()
    print(len(congestion_kpis))

    congestion_kpis.index = congestion_kpis.site_id
  
    del congestion_kpis["site_id"]
    congestion_kpis.to_parquet(congestion_kpis_output.path)