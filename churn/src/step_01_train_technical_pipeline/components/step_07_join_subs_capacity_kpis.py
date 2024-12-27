from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def join_subs_capacity_kpis(input_data_subs: Input[Dataset], input_congestion_kpis: Input[Dataset], output_subs_congestion_kpis: Output[Dataset]
            ):
    
    import pandas as pd
    
    data_subs = pd.read_parquet(input_data_subs.path)
    congestion_kpis = pd.read_parquet(input_congestion_kpis.path)

    data_subs = data_subs.join(congestion_kpis, on="MAIN_SITE_CODE_NEW")
    print(data_subs[["SUBS_STATUS_DESC", "MAIN_SITE_CODE_NEW"]].groupby("SUBS_STATUS_DESC").count())

    data_subs = data_subs.dropna()
    print(data_subs[["SUBS_STATUS_DESC", "MAIN_SITE_CODE_NEW"]].groupby("SUBS_STATUS_DESC").count())

    data_subs = data_subs.drop_duplicates()
    
    data_subs.to_parquet(output_subs_congestion_kpis.path)