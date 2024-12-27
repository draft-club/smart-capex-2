from kfp.dsl import Dataset, Output, component
from utils.config import pipeline_config


@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def get_data_subs(project_id:str, location:str, m0: int, data_subs_output: Output[Dataset]
            ):
    
    import pandas as pd
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id, location=location)
    
    m1 = (m0 - 1) % 12
    query_m0_subs = """select distinct SUBS_ID, SUBS_STATUS_DESC, MAIN_SITE_CODE_NEW from `oro-smart-capex-001-dev.smart_capex_raw.v_users_usage` where PRODUCT_LINE_CODE = "POSTPAID" and EXTRACT(month from DATE_ID) =""" + str(m0) + """ and SUBS_ID in (select SUBS_ID from `oro-smart-capex-001-dev.smart_capex_raw.v_users_usage` where EXTRACT(month from DATE_ID) =""" + str(m1) + """ and SUBS_STATUS_DESC = "Active Subscriber")"""

    data_subs = client.query(query_m0_subs).to_dataframe()
    data_subs = data_subs.drop_duplicates()
    data_subs.index = data_subs.SUBS_ID
    del data_subs['SUBS_ID']
    data_subs = data_subs[data_subs.SUBS_STATUS_DESC.isin(["Switched Off", "Active Subscriber"])]
    data_subs = data_subs.dropna()
    print("data_subs", data_subs.shape)
    data_subs.to_parquet(data_subs_output.path)
    