from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def get_data_kpis(project_id:str, location:str, m0: int, delta: int, kpis_output: Output[Dataset]
            ):
    
    import pandas as pd
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id, location=location)
    m = (m0 - delta) % 12
    
    query_kpis = """select distinct SUBS_ID, SMS_SENT_CNT, SMS_RECEIVED_CNT, CALLS_RECEIVED_SEC, CALLS_RECEIVED_SEC, CALLS_MADE_SEC, DATA_TRAFFIC_KB, INT_CALL_COST_MIN, CALLS_MADE_OFFNET_CNT, CALLS_RECEIVED_OFFNET_CNT, FREE_SERVICES_CNT, OVERBUNDLE_AMT_EUR, MAIN_SITE_CODE_NEW from `oro-smart-capex-001-dev.smart_capex_raw.v_users_usage` where EXTRACT(month from DATE_ID) =""" + str(m) + """ and SUBS_STATUS_DESC = "Active Subscriber" and PRODUCT_LINE_CODE = "POSTPAID" """
    kpis = client.query(query_kpis).to_dataframe()
    kpis = kpis.dropna()
    kpis = kpis.drop_duplicates()
    
    kpis.index = kpis.SUBS_ID
    del kpis['SUBS_ID']
    
    kpis = kpis.rename({"MAIN_SITE_CODE_NEW": "MAIN_SITE_CODE_NEW_m" + str(delta)}, axis =1)
    print(kpis.head())
    kpis.to_parquet(kpis_output.path)