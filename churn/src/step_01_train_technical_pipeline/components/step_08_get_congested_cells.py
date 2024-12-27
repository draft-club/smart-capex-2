from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def get_congested_cells(project_id:str, location:str, m0: int, delta:int , congested_cells: Output[Dataset]):
    
    import pandas as pd
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id, location=location)
    m = (m0 - delta) % 12
    
    query_nb_cell_cong = """SELECT site_id, month, count(cell_name) as nb_cell_cong FROM `oro-smart-capex-001-dev.intermediate_results.data_preparation_oss_counter_agg_t` where average_number_of_users_in_queue > 2.5 and average_throughput_user_dl < 3000 and year= 2024 and month = """ + str(m) + """ group by site_id, month order by site_id, month ASC"""
    nb_cell_cong = client.query(query_nb_cell_cong).to_dataframe().drop_duplicates()
    nb_cell_cong.index = nb_cell_cong.site_id
    nb_cell_cong = nb_cell_cong[['nb_cell_cong']].fillna(0)

    nb_cell_cong.to_parquet(congested_cells.path)