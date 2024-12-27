from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def join_kpis(input_m1_kpis: Input[Dataset], input_m3_kpis: Input[Dataset], output_kpis: Output[Dataset]
            ):
    
    import pandas as pd

    m1_kpis = pd.read_parquet(input_m1_kpis.path)

    m3_kpis = pd.read_parquet(input_m3_kpis.path)

    data_kpis = m1_kpis.join(m3_kpis, rsuffix = "_m3")
    data_kpis = data_kpis.dropna()
    data_kpis = data_kpis.drop_duplicates()
    data_kpis = data_kpis[data_kpis.MAIN_SITE_CODE_NEW_m1 == data_kpis.MAIN_SITE_CODE_NEW_m3]
    
    data_kpis.to_parquet(output_kpis.path)