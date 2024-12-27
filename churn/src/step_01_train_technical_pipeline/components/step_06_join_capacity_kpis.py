from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def join_capacity_kpis(input_m1_congestion_kpis: Input[Dataset], input_m3_congestion_kpis: Input[Dataset], output_congestion_kpis: Output[Dataset]
            ):
    
    import pandas as pd

    m1_congestion_kpis = pd.read_parquet(input_m1_congestion_kpis.path)

    m3_congestion_kpis = pd.read_parquet(input_m3_congestion_kpis.path)

    congestion_kpis = m1_congestion_kpis.join(m3_congestion_kpis, rsuffix = "_m3")
    print(len(congestion_kpis))
    congestion_kpis = congestion_kpis.dropna()
    print(len(congestion_kpis))
    congestion_kpis = congestion_kpis.drop_duplicates()

    congestion_kpis['average_nb_user_in_queue_diff'] = congestion_kpis["average_nb_user_in_queue"] - congestion_kpis["average_nb_user_in_queue_m3"]
    congestion_kpis['average_throughput_user_dl_diff'] = congestion_kpis["average_throughput_user_dl"] - congestion_kpis["average_throughput_user_dl_m3"]
    congestion_kpis['max_nb_user_in_queue_diff'] = congestion_kpis["max_nb_user_in_queue"] - congestion_kpis["max_nb_user_in_queue_m3"]
    congestion_kpis['min_throughput_user_dl_diff'] = congestion_kpis["min_throughput_user_dl"] - congestion_kpis["min_throughput_user_dl_m3"]
    congestion_kpis.to_parquet(output_congestion_kpis.path)