from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def join_subs_kpis(input_data_subs: Input[Dataset], input_kpis: Input[Dataset], subs_kpis: Output[Dataset]
            ):
    
    import pandas as pd
    
    data_subs = pd.read_parquet(input_data_subs.path)
    data_kpis = pd.read_parquet(input_kpis.path)

    data_subs = data_subs.join(data_kpis).dropna().drop_duplicates()

    data_subs = data_subs[data_subs.MAIN_SITE_CODE_NEW == data_subs.MAIN_SITE_CODE_NEW_m3]
    del data_subs["MAIN_SITE_CODE_NEW_m3"]
    
    features =  ["SMS_SENT_CNT", "SMS_RECEIVED_CNT", "CALLS_RECEIVED_SEC", "CALLS_RECEIVED_SEC", "CALLS_MADE_SEC", "DATA_TRAFFIC_KB", "INT_CALL_COST_MIN", "CALLS_MADE_OFFNET_CNT", "CALLS_RECEIVED_OFFNET_CNT", "FREE_SERVICES_CNT", "OVERBUNDLE_AMT_EUR"]
    features_m3 = [x + "_m3" for x in features]
    features_diff = [x + "_diff" for x in features]

    for i, f in enumerate(features_diff): 
        data_subs[f] = data_subs[features_m3[i]] - data_subs[features[i]]
    data_subs.to_parquet(subs_kpis.path)