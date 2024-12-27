# IMPORT THE REQUIRED LIBRARIES
import os
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Output,
                        Model,
                        Metrics,
                        Markdown,
                        HTML,
                        component, 
                        OutputPath, 
                        InputPath)

from kfp.v2 import compiler
from google.cloud import aiplatform as vertex_
from google.cloud.aiplatform import pipeline_jobs

from datetime import datetime
import pandas as pd
import subprocess


# project = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
# PROJECT_ID = project.stdout
PROJECT_ID = "oro-smart-capex-001-dev"
# PROJECT_ID = PROJECT_ID.replace('\n', '')
REGION = 'europe-west3'

BUCKET_NAME="gs://"+"oro-smart-capex-config-001-dev/"
PIPELINE_ROOT = f"{BUCKET_NAME}/smartcapex_pipeline/"


# Custom base image created using docker

IMAGE_NAME = "smartcapex-pipeline"
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/smart-capex-capacity/{IMAGE_NAME}"



sa=
# # Google credentials should be assigned to a service account
# # Also note that resources should be configured to use the proper, restricted service account

 #Decocde the Base64 encoded Google credentials and write the m to a file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/workspaces/smart-capex-capacity/templates/oro-smart-capex-001-dev-sa-oro-data-cicd-capex-dev.json'
with open('src/google-credentials.json', 'w') as file:
   file.write(sa)








os.environ['HTTP_PROXY'] = 'http://proxybkp.si.francetelecom.fr:8080'


#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAib3NuLXNtYXJ0Y2FwZXgtNDA0LXNieCIsCiAgInByaXZhdGVfa2V5X2lkIjogIjg0MGU1YmRkZjI3MTY5ZjUwZTljMWQ0NGQ4MjcyZTRkYWE4YWY5MDAiLAogICJwcml2YXRlX2tleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2UUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktjd2dnU2pBZ0VBQW9JQkFRQzM4OUdUNkx4aWZ0SGJcbnZuOGdhUXdQaEVBbGJ1K3ZveXd4K0wwL3dia3lLeURmKzQ1L2FPR3R5QnZKVk9USFIzNmRRLzNkOXMzcUM5eTFcbnNyNWtGZTE1bEdIcndiVjQ1TVNoNEpFRklKVXFmd0hod0Rka0h2SGZQM1ZCWkMzaGVsNzdPVDBmRm1yK2wveE9cbnUrMjVzL3E2VksrMktXcmI2ZkNZdEk2ZnkvTXpNdmh3UExuc29CVlhwSUNPUnlmT2RGZ1prRFRiMUhzUjRTYWlcblA2eUo4eCtJM2ZaNXgyb0FFL2dtNVdZN3dTTDIvSGw5cjlmSVJuV3o5YTZMOXdFNUdrV2R4SStFdVVZSkxmR2FcbmZHZzhhRnpxdW52MGZSZW5FNGVxL3dvNkdLbWxPbkMvZ1FMZDdHQWxneEExUFRzd3MxZCsrNVcxaDdNRStVUGJcbmkrN1UzQWlwQWdNQkFBRUNnZ0VBRW5UWXFNWDNyTXc3b3hnY2FqU1AvTGlRdFJESjVLM3kvN3FNeG1wWitnSWdcbnhNaFpGa2lBS0M4Rk5yQ1pXYk0vTFdsR2ZmQkczYjQ5OFU4VFVqaHZBTDFXQzlEK003aUVVQmVYaFZPdEU3djFcbmhPQ0R6anhhbVVNbE1SL3JpSTErSFNlempieHlsTVBNRVZ4Q3NGaXlCZUw4VkVNWWtTTUtQRkFWWlV3UnhTZWlcbmczUXRwenFtV0Q0ZUJEVFlWa2tFR0lZSDl4UDBVT2RjOGFNK3laYTV6K2FOQ1RkeWxLc1ljTTF3Wi8rWG4xSnRcbkFpaEkyVVVnWDg5b2ZsUEFLTWo2V2NEUkZTY2lCcDFuZVAvV25xaUU1K2NtaCtVZ1BOLzRiTng1cSt6NThxSlJcbjEzRGxiemhmeW94MkVkMHh1VTBUTmNWcXRBNld5cWtUVm84SnM0dzNXUUtCZ1FEOEU4YlVzQjl0NVRwSWJSb2RcbkNPMWFMYXlCbUVCWG9kbXZXR0IzMjZ1TEtJRExyNVNJdGRoUkF5MWx2RmlDcmt1OWNjWGl6ZCtzRkVmNWtBZnBcbnZhRGdMMk9lKzIvVG1Eck9WVTdYNllvVGRQM25jZ216UmU3dW4yWEoySjloaEVDQkQyUlN4SUVtSUl5c3VtT0pcbmx4Yi9iYmpxaHlIdHRERHRUUWc1Uk5tY1pRS0JnUUM2MEtXVlhzQlRQWWJZRFNHVEpZOU8veUwzUVZMZlZMWUxcbmR4cWVtbE9FSi9uR2MxeFUyelhkLzg3c3pPSmM3eW5YSStlQkJBN0dlUUgyM2pnU0w2WGtBc2hQdnlRNGdBbnRcbjVjMExzMXRhMDZpck5KUnd4NGJ0N0t1MFJGTzhnMDhrQ3JmTXB0UXJJWGg2QWxNb3o2MGFubDJ3VWtNcWJza0FcbkFUYlR1MWdzOVFLQmdHVnBsMjV0eE5jemgzVW4zMythM2RLUDJYenh3Y0QvcmxJcTNmU2FmYk1vZ2xodnRQUWpcbkpIbkRLM3BvZ2J0aFg3dEJrSGtrbGozbWt1WkdHY2pocjExQjgzUThkOHJLemEzQkNFMDQrWUhHYVhlNW0wbHlcbmN5T3hJUVJKa0NWdFRYNGVzUi9UU3BvS01rNHpWbVErVXRSRVVrYVlRd0FjcENweitVRUJBQU01QW9HQUh1Tm9cbnZXM1JOdkl4WFgrdVVYb2dXOXRybUo1QWFaVEVGTms0bVlqQ3psTWR4V1pGbWZJMDBlUDkvc0ZSbkRRZkl1ZFlcbjI4Z2orVVVBd2lTeitLM1FMQWNadjdYRzgyQ1lRN0YvV2JQcUl2WmtLUXFra0pFdENpSGJzZzZxR2IxTVZKVkJcbkZxRnU2MEs1Zk5MdGxRM2hmVWs2REhGTmtiS0hvV3lSK0NnOXlCRUNnWUVBcUw3ZnVVdlcrbmQ1cHVNYzMyS1ZcbmVnajcyMTdwaGJHaHMyQ09LZ3N4NnFlQUY5TDk1QzlKaytIeDNjVitGSTVEbnBNalk1blJkTmV6OUU1b21DUnhcbmRsMHBoWUpnaGplaW9KRDJYRUJ1ZU4zZjFDamltTkVWM2NXdEliREtWa2U4N2lqUXRwdzlUNUZOTERPbjl6ckdcbk1HdkhhZndneGtRQkQ5bmp5aXJ4Zk5rPVxuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLVxuIiwKICAiY2xpZW50X2VtYWlsIjogIjk4OTU0NDk1MTM0OC1jb21wdXRlQGRldmVsb3Blci5nc2VydmljZWFjY291bnQuY29tIiwKICAiY2xpZW50X2lkIjogIjEwMDQwODcxMjExNDY5MzI2NDcxOSIsCiAgImF1dGhfdXJpIjogImh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi9hdXRoIiwKICAidG9rZW5fdXJpIjogImh0dHBzOi8vb2F1dGgyLmdvb2dsZWFwaXMuY29tL3Rva2VuIiwKICAiYXV0aF9wcm92aWRlcl94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL29hdXRoMi92MS9jZXJ0cyIsCiAgImNsaWVudF94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL3JvYm90L3YxL21ldGFkYXRhL3g1MDkvOTg5NTQ0OTUxMzQ4LWNvbXB1dGUlNDBkZXZlbG9wZXIuZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgInVuaXZlcnNlX2RvbWFpbiI6ICJnb29nbGVhcGlzLmNvbSIKfQ=='



vertex_.init(project=PROJECT_ID, location=REGION)


@component(
    base_image=BASE_IMAGE,
)

def get_data(
    project_id: str, 
    dataset_src: str,
    table_id: str,
    dataset_raw: Output[Dataset],
):
    """
    Get data from BigQuery
    """
    
    from google.cloud import bigquery 
    import pandas as pd
    
    client = bigquery.Client(project=project_id)
    query = f"select * from `{project_id}.{dataset_src}.{table_id}`"
    data = client.query(query = query).to_dataframe()
    data.to_csv(dataset_raw.path, index=False)
    print(f"Data successfully read from BigQuery table : {project_id}.{dataset_src}.{table_id}")
    

    
@component(
    base_image=BASE_IMAGE,
)

def preprocess_data(
    raw_sites_df: Input[Dataset],
    raw_oss_2g_df: Input[Dataset],
    raw_oss_3g_df: Input[Dataset],
    raw_oss_4g_df: Input[Dataset],
    raw_oss_rtx_df: Input[Dataset],
    dataset_sites_preprocessed: Output[Dataset],
    dataset_oss_preprocessed: Output[Dataset],
):
    """
    preprocess data
    """
    
    
    import pandas as pd
    from src.d02_preprocessing.process_sites import sites_preprocessing
    from src.d02_preprocessing.process_oss import preprocessing_oss_counter_weekly
   
    raw_sites_df = pd.read_csv(raw_sites_df.path)
    raw_oss_2g_df = pd.read_csv(raw_oss_2g_df.path)
    raw_oss_3g_df = pd.read_csv(raw_oss_3g_df.path)
    raw_oss_4g_df = pd.read_csv(raw_oss_4g_df.path)
    raw_oss_rtx_df = pd.read_csv(raw_oss_rtx_df.path)
    
    
    sites_preprocessed = sites_preprocessing(raw_sites_df)
    oss_preprocessed = preprocessing_oss_counter_weekly(sites_preprocessed,raw_oss_2g_df,raw_oss_3g_df,raw_oss_4g_df,raw_oss_rtx_df)

    oss_preprocessed.drop('site_id', axis=1, inplace=True)
    oss_preprocessed = oss_preprocessed.merge(sites_preprocessed[['site_id', 'cell_name']].drop_duplicates(), how='left', on='cell_name')
    oss_preprocessed.dropna(subset=['site_id'], inplace=True)
    oss_preprocessed.rename(columns={'total_data_traffic_up_gb':'total_data_traffic_ul_gb'},inplace=True)

    sites_preprocessed.to_csv(dataset_sites_preprocessed.path, index=False)
    oss_preprocessed.to_csv(dataset_oss_preprocessed.path, index=False)
    
@component(
    base_image=BASE_IMAGE,
)

def compute_sites_distances(
    dataset_sites_preprocessed: Input[Dataset],
    df_distance: Output[Dataset],

):
    """
    Process distances between sites
    """
    
    import pandas as pd
    from src.d03_capacity.technical_modules.cluster_selection import compute_distance_between_sites
   
    df_sites = pd.read_csv(dataset_sites_preprocessed.path)
    print("////////////////////df_sites from compute_sites_distances/////////")
    print(df_sites.shape)

    distance = compute_distance_between_sites(df_sites)
    
    distance.to_csv(df_distance.path, index=False)

    
@component(
    base_image=BASE_IMAGE,
)

def save_to_bigquery(
    dataset_preprocessed: Input[Dataset],
    project_id: str,
    dataset_id: str,
    table_output: str
):
    
    """
    save data to bq
    """
    import pandas as pd
    import pandas_gbq
    
    dataset_preprocessed = pd.read_csv(dataset_preprocessed.path)
    
    pandas_gbq.to_gbq(dataset_preprocessed, f'{project_id}.{dataset_id}.{table_output}', project_id=project_id, if_exists='replace')
    
@component(
    base_image=BASE_IMAGE,
)

def traffic_forcasting(
    dataset_preprocessed: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    df_traffic_predicted: Output[Dataset]
    

):
    
    """
    Process traffic forecasting
    """
    
    import pandas as pd
    import pandas_gbq
    from src.d03_capacity.technical_modules.traffic_forecasting import prophet
    
    traffic_preprocessed = pd.read_csv(dataset_preprocessed.path)
    site_preprocessed = pd.read_csv(dataset_site_preprocessed.path)
    
    traffic_predicted = prophet(traffic_preprocessed)
    
    traffic_predicted.drop('site_id', axis=1, inplace=True)
    traffic_predicted = traffic_predicted.merge(site_preprocessed[['site_id', 'cell_name']].drop_duplicates(),
                                                      how='left',
                                                      on='cell_name')
    traffic_predicted.dropna(subset=['site_id'], inplace=True)
    
    traffic_predicted.to_csv(df_traffic_predicted.path, index=False)
    
@component(
    base_image=BASE_IMAGE,
)

def process_bands_to_upgrade(
    dataset_distance: Input[Dataset],
    df_traffic_predicted: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    dataset_typology_sector: Input[Dataset],
    dataset_selected_band_per_site : Output[Dataset],
    dataset_affected_cells: Output[Dataset],
    dataset_cluster_future_upgrades: Output[Dataset]
  
):
    
    """
    Build features related to bands upgrade
    """
    
    import pandas as pd
    import pandas_gbq
    from src.d03_capacity.technical_modules.upgrade_selection import upgrade_selection_pipeline
    from src.d03_capacity.technical_modules.cluster_selection import get_cluster_of_future_upgrades
    from src.d00_conf.conf import conf, conf_loader
    
    conf_loader("OSN")

    
    df_predicted_traffic_kpis = pd.read_csv(df_traffic_predicted.path)
    df_sites = pd.read_csv(dataset_site_preprocessed.path)
    df_typology_sector = pd.read_csv(dataset_typology_sector.path)
    df_distance = pd.read_csv(dataset_distance.path)
    print("Process bands to upgrade")
    
    selected_band_per_site = upgrade_selection_pipeline(df_predicted_traffic_kpis.copy(),df_typology_sector,df_sites)
    
    df_affected_cells = df_predicted_traffic_kpis.merge(selected_band_per_site,
                                                              on='site_id',
                                                              how='inner')
    
    df_cluster_future_upgrades = get_cluster_of_future_upgrades(selected_band_per_site,
                                                                df_distance,
                                                                max_neighbors=conf['TRAFFIC_IMPROVEMENT'][
                                                                    'MAX_NUMBER_OF_NEIGHBORS'])
    
    selected_band_per_site.to_csv(dataset_selected_band_per_site.path, index=False)
    df_affected_cells.to_csv(dataset_affected_cells.path, index=False)
    df_cluster_future_upgrades.to_csv(dataset_cluster_future_upgrades.path, index=False)

 
    
@component(
    base_image=BASE_IMAGE,
)
def train_technical_pipeline (
    dataset_oss_preprocessed: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    dataset_distance: Input[Dataset],
    dataset_cell_affected: Output[Dataset],
    dataset_list_of_upgrades: Output[Dataset],
    dataset_sites_to_remove: Output[Dataset],
    dataset_data_dl_traffic_features: Output[Dataset],
    dataset_data_ul_traffic_features: Output[Dataset],
    dataset_voice_traffic_features: Output[Dataset],
    output_data_dl_model: Output[Model],
    output_data_ul_model: Output[Model],
    output_voice_model: Output[Model],
    dataset_traffic_by_region: Output[Dataset],
    output_linear_regression_data_dl: Output[Model],
    output_linear_regression_data_ul: Output[Model],
    output_linear_regression_voice: Output[Model]
):
    
    """
    Build features for traffic improvement & trend models and train them
    """
        
    import pandas as pd
    from src.d03_capacity.technical_modules.activation_model import get_affected_cells_with_interactions_between_upgrades
    from src.d03_capacity.technical_modules.cluster_selection import get_cluster_of_affected_sites
    from src.d03_capacity.technical_modules.traffic_improvement import get_all_traffic_improvement_features, compute_traffic_after, train_traffic_improvement_model
    from src.d03_capacity.technical_modules.traffic_improvement_trend import compute_traffic_by_region, train_trend_model_with_linear_regression
    from src.d00_conf.conf import conf, conf_loader
    import pickle

    conf_loader("OSN")

    df_traffic_weekly_kpis = pd.read_csv(dataset_oss_preprocessed.path)
    df_sites = pd.read_csv(dataset_site_preprocessed.path)
    df_distance = pd.read_csv(dataset_distance.path)

    print("train_technical_pipeline")
    df_traffic_weekly_kpis = df_traffic_weekly_kpis.replace({'cell_band': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                                           'F1_U2100':'U2100','F4_U2100':'U2100','F2_U900':'U900'}})
    
    df_cell_affected = get_affected_cells_with_interactions_between_upgrades(df_traffic_weekly_kpis)
    
    list_of_upgrades, sites_to_remove = get_cluster_of_affected_sites(df_cell_affected,
                                                                    df_distance,
                                                                    max_neighbors=
                                                                    conf['TRAFFIC_IMPROVEMENT'][
                                                                    'MAX_NUMBER_OF_NEIGHBORS'])
    

    df_data_dl_traffic_features = get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                                                           df_cell_affected,
                                                                           list_of_upgrades,
                                                                           sites_to_remove,
                                                                           type_of_traffic='data_dl',
                                                                           kpi_to_compute_upgrade_effect=[
                                                                           "total_data_traffic_dl_gb"],
                                                                            upgraded_to_not_consider=[])
    df_data_dl_traffic_features = compute_traffic_after(df_data_dl_traffic_features, df_traffic_weekly_kpis,'total_voice_traffic_kerlands')
    df_data_ul_traffic_features = get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                                                           df_cell_affected,
                                                                           list_of_upgrades,
                                                                           sites_to_remove,
                                                                           type_of_traffic='data_ul',
                                                                           kpi_to_compute_upgrade_effect=[
                                                                           "total_data_traffic_ul_gb"],
                                                                            upgraded_to_not_consider=[])
    df_data_ul_traffic_features = compute_traffic_after(df_data_ul_traffic_features, df_traffic_weekly_kpis,'total_voice_traffic_kerlands')
    df_voice_traffic_features = get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                                                           df_cell_affected,
                                                                           list_of_upgrades,
                                                                           sites_to_remove,
                                                                           type_of_traffic='voice',
                                                                           kpi_to_compute_upgrade_effect=[
                                                                           "total_voice_traffic_kerlands"],
                                                                            upgraded_to_not_consider=[])
    df_voice_traffic_features = compute_traffic_after(df_voice_traffic_features, df_traffic_weekly_kpis,'total_voice_traffic_kerlands')

                                                                    
    model_rf_data_dl = train_traffic_improvement_model(df_data_dl_traffic_features,
                                    type_of_traffic='data_dl',
                                    remove_samples_with_target_variable_lower=True,
                                    bands_to_consider=['G900', 'G1800','L2600', 'L1800', 'L800', 'U2100', 'U900'])
    model_rf_data_ul = train_traffic_improvement_model(df_data_ul_traffic_features,
                                    type_of_traffic='data_ul',
                                    remove_samples_with_target_variable_lower=True,
                                    bands_to_consider=['G900', 'G1800','L2600', 'L1800', 'L800', 'U2100', 'U900'])
    model_rf_voice = train_traffic_improvement_model(df_voice_traffic_features,
                                    type_of_traffic='voice',
                                    remove_samples_with_target_variable_lower=True,
                                    bands_to_consider=['G900', 'G1800','L2600', 'L1800', 'L800', 'U2100', 'U900'])

    
    df_traffic_by_region = compute_traffic_by_region(df_sites,
                                                 df_traffic_weekly_kpis,
                                                 kpi_to_compute_trend=[
                                                        'total_data_traffic_dl_gb',
                                                        'total_data_traffic_ul_gb',
                                                        'total_voice_traffic_kerlands'])

    series_model_lr_data_dl = train_trend_model_with_linear_regression(df_traffic_by_region,
                                             variable_to_group_by=['site_region'],
                                             kpi_to_compute_trend=['total_data_traffic_dl_gb'])
    series_model_lr_data_ul = train_trend_model_with_linear_regression(df_traffic_by_region,
                                             variable_to_group_by=['site_region'],
                                             kpi_to_compute_trend=['total_data_traffic_ul_gb'])
    series_model_lr_voice = train_trend_model_with_linear_regression(df_traffic_by_region,
                                             variable_to_group_by=['site_region'],
                                             kpi_to_compute_trend=['total_voice_traffic_kerlands'])
    
 
    with open(output_data_dl_model.path, 'wb') as file: 
        pickle.dump({"model": model_rf_data_dl}, file)

    with open(output_data_ul_model.path, 'wb') as file: 
        pickle.dump({"model": model_rf_data_ul}, file)
    
    with open(output_voice_model.path, 'wb') as file: 
        pickle.dump({"model": model_rf_voice}, file) 
       
        
    model_lr_data_dl = series_model_lr_data_dl[0]
    with open(output_linear_regression_data_dl.path, 'wb') as file: 
        pickle.dump({"model": model_lr_data_dl}, file)

    model_lr_data_ul = series_model_lr_data_ul[0]
    with open(output_linear_regression_data_ul.path, 'wb') as file: 
        pickle.dump({"model": model_lr_data_ul}, file)

    model_lr_voice = series_model_lr_voice[0]
    with open(output_linear_regression_voice.path, 'wb') as file: 
        pickle.dump({"model": model_lr_voice}, file)
                
        
    df_cell_affected.to_csv(dataset_cell_affected.path, index=False)
    list_of_upgrades.to_csv(dataset_list_of_upgrades.path, index=False)
    sites_to_remove = pd.DataFrame(sites_to_remove)
    sites_to_remove.to_csv(dataset_sites_to_remove.path, index=False)
    df_data_dl_traffic_features.to_csv(dataset_data_dl_traffic_features.path, index=False)
    df_data_ul_traffic_features.to_csv(dataset_data_ul_traffic_features.path, index=False)
    df_voice_traffic_features.to_csv(dataset_voice_traffic_features.path, index=False)
    df_traffic_by_region.to_csv(dataset_traffic_by_region.path, index=False)
    
@component(
base_image=BASE_IMAGE,
)
def process_traffic_improvement (
    df_traffic_predicted: Input[Dataset],
    dataset_affected_cells: Input[Dataset],
    dataset_cluster_future_upgrades: Input[Dataset],
    dataset_selected_band_per_site: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    output_data_dl_model : Input[Model],
    output_data_ul_model : Input[Model],
    output_voice_model : Input[Model],
    output_linear_regression_data_dl: Input[Model],
    output_linear_regression_data_ul: Input[Model],
    output_linear_regression_voice: Input[Model],  
    dataset_traffic_features_future_upgrades_data_dl: Output[Dataset],
    dataset_traffic_features_future_upgrades_data_ul: Output[Dataset],
    dataset_traffic_features_future_upgrades_voice: Output[Dataset], 
    dataset_traffic_features_future_upgrades_prediction_data_dl: Output[Dataset],
    dataset_traffic_features_future_upgrades_prediction_data_ul: Output[Dataset],
    dataset_traffic_features_future_upgrades_prediction_voice: Output[Dataset],
    dataset_increase_traffic_after_upgrade_data_dl: Output[Dataset],
    dataset_increase_traffic_after_upgrade_data_ul: Output[Dataset],
    dataset_increase_traffic_after_upgrade_voice: Output[Dataset],
    dataset_predicted_increase_in_traffic_by_the_upgrade: Output[Dataset],
    
): 
    """
    Predict traffic improvement and trend after upgrade
    """
   
    import pandas as pd
    from src.d03_capacity.technical_modules.traffic_improvement import get_all_traffic_improvement_features, compute_traffic_after, predict_traffic_improvement_model, get_traffic_improvement   
    from src.d03_capacity.technical_modules.traffic_improvement_trend import predict_improvement_traffic_trend_kpis, merge_predicted_improvement_traffics
    from src.d00_conf.conf import conf, conf_loader
    import pickle
    conf_loader("OSN")
    

    print("------------------start process_traffic_improvement----------------- ")
    df_predicted_traffic_kpis = pd.read_csv(df_traffic_predicted.path)
    print("$$$$$$$$$$$$$$$$$$$$$$df_predicted_traffic_kpis.shape$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(df_predicted_traffic_kpis.shape)
    df_future_cell_affected = pd.read_csv(dataset_affected_cells.path)
    print("***************df_future_cell_affected.shape***********************")
    print(df_future_cell_affected.shape)
    df_cluster_future_upgrades = pd.read_csv(dataset_cluster_future_upgrades.path)
    print("888888888888888888df_cluster_future_upgrades888888888888888888888")
    print(df_cluster_future_upgrades.shape)


    selected_band_per_site = pd.read_csv(dataset_selected_band_per_site.path)
    print("99999999999999999selected_band_per_site99999999999999999999")
    print(selected_band_per_site.shape)



    df_sites = pd.read_csv(dataset_site_preprocessed.path)
    print("10101010101010101010df_sites10101010101010101010")
    print(df_sites.shape)
    
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.replace(
            {'cell_band': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                           'F1_U2100': 'U2100', 'F4_U2100': 'U2100', 'F2_U900': 'U900'}})

    selected_band_per_site = selected_band_per_site.replace(
            {'bands_upgraded': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                           'F1_U2100': 'U2100', 'F4_U2100': 'U2100', 'F2_U900': 'U900'}})


    df_traffic_features_future_upgrades_data_dl = get_all_traffic_improvement_features(df_predicted_traffic_kpis,
                                                                 df_future_cell_affected,
                                                                 df_cluster_future_upgrades,
                                                                 sites_to_remove = [],
                                                                 type_of_traffic='data_dl',
                                                                 group_bands=False,
                                                                 remove_sites_with_more_than_one_upgrade_same_cluster=False,
                                                                 kpi_to_compute_upgrade_effect=["total_data_traffic_dl_gb"],
                                                                 capacity_kpis_features=["cell_occupation_dl_percentage"],
                                                                 upgraded_to_not_consider=[],
                                                                 compute_target_variable_traffic = False)              
    df_traffic_features_future_upgrades_data_dl = compute_traffic_after(df_traffic_features_future_upgrades_data_dl, df_predicted_traffic_kpis,'total_data_traffic_dl_gb')

    df_traffic_features_future_upgrades_data_ul = get_all_traffic_improvement_features(df_predicted_traffic_kpis,
                                                                 df_future_cell_affected,
                                                                 df_cluster_future_upgrades,
                                                                 sites_to_remove = [],
                                                                 type_of_traffic='data_ul',
                                                                 group_bands=False,
                                                                 remove_sites_with_more_than_one_upgrade_same_cluster=False,
                                                                 kpi_to_compute_upgrade_effect=["total_data_traffic_ul_gb"],
                                                                 capacity_kpis_features=["cell_occupation_dl_percentage"],
                                                                 upgraded_to_not_consider=[],
                                                                 compute_target_variable_traffic = False)              
    df_traffic_features_future_upgrades_data_ul = compute_traffic_after(df_traffic_features_future_upgrades_data_ul, df_predicted_traffic_kpis,'total_data_traffic_ul_gb')

    df_traffic_features_future_upgrades_voice = get_all_traffic_improvement_features(df_predicted_traffic_kpis,
                                                                 df_future_cell_affected,
                                                                 df_cluster_future_upgrades,
                                                                 sites_to_remove = [],
                                                                 type_of_traffic='voice',
                                                                 group_bands=False,
                                                                 remove_sites_with_more_than_one_upgrade_same_cluster=False,
                                                                 kpi_to_compute_upgrade_effect=["total_voice_traffic_kerlands"],
                                                                 capacity_kpis_features=["cell_occupation_dl_percentage"],
                                                                 upgraded_to_not_consider=[],
                                                                 compute_target_variable_traffic = False)              
    df_traffic_features_future_upgrades_voice = compute_traffic_after(df_traffic_features_future_upgrades_voice, df_predicted_traffic_kpis,'total_voice_traffic_kerlands')
    
    

        
    #file_name_data = model_rf_data.path
    file_name_data = output_data_dl_model.path
    with open(file_name_data, "rb") as file:
        model = pickle.load(file)        
    model_rf = model["model"] 

    df_traffic_features_future_upgrades_prediction_data_dl = predict_traffic_improvement_model(df_traffic_features_future_upgrades_data_dl, model_rf, type_of_traffic='data_dl')

    file_name_data = output_data_ul_model.path
    with open(file_name_data, "rb") as file:
        model = pickle.load(file)        
    model_rf = model["model"] 

    df_traffic_features_future_upgrades_prediction_data_ul = predict_traffic_improvement_model(df_traffic_features_future_upgrades_data_ul, model_rf, type_of_traffic='data_ul')

    file_name_data = output_voice_model.path
    with open(file_name_data, "rb") as file:
        model = pickle.load(file)        
    model_rf = model["model"] 

    df_traffic_features_future_upgrades_prediction_voice = predict_traffic_improvement_model(df_traffic_features_future_upgrades_voice, model_rf,
                                                                        type_of_traffic='voice') 
    

     
    df_increase_traffic_after_upgrade_data_dl = get_traffic_improvement(df_predicted_traffic_kpis,
                                                          selected_band_per_site,
                                                          df_traffic_features_future_upgrades_prediction_data_dl,
                                                          kpi_to_compute_upgrade_effect=["total_data_traffic_dl_gb"])
    
    df_increase_traffic_after_upgrade_data_ul = get_traffic_improvement(df_predicted_traffic_kpis,
                                                          selected_band_per_site,
                                                          df_traffic_features_future_upgrades_prediction_data_ul,
                                                          kpi_to_compute_upgrade_effect=["total_data_traffic_ul_gb"])
    
    df_increase_traffic_after_upgrade_voice = get_traffic_improvement(df_predicted_traffic_kpis,
                                                          selected_band_per_site,
                                                          df_traffic_features_future_upgrades_prediction_voice,
                                                          kpi_to_compute_upgrade_effect=[
                                                              "total_voice_traffic_kerlands"])

    
    file_name_data = output_linear_regression_data_dl.path
    with open(file_name_data, "rb") as file:
        model = pickle.load(file)        
    model_lr = model["model"]   
    
    data_dl = predict_improvement_traffic_trend_kpis(df_increase_traffic_after_upgrade_data_dl,
                                                                                df_sites,
                                                                                model_lr,
                                                                                variable_to_group_by=['site_region'],
                                                                                max_yearly_increment=50,
                                                                                kpi_to_compute_trend=[
                                                                                    'total_data_traffic_dl_gb'],
                                                                                max_weeks_to_predict=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_PREDICT'],
                                                                                max_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                min_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MIN_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                weeks_to_wait_after_the_upgrade=
                                                                                conf['TRAFFIC_IMPROVEMENT'][
                                                                                    'WEEKS_TO_WAIT_AFTER_UPGRADE'])
    

    file_name_data = output_linear_regression_data_ul.path
    with open(file_name_data, "rb") as file:
        model = pickle.load(file)        
    model_lr = model["model"]   
    
    data_ul = predict_improvement_traffic_trend_kpis(df_increase_traffic_after_upgrade_data_ul,
                                                                                df_sites,
                                                                                model_lr,
                                                                                variable_to_group_by=['site_region'],
                                                                                max_yearly_increment=50,
                                                                                kpi_to_compute_trend=[
                                                                                    'total_data_traffic_ul_gb'],
                                                                                max_weeks_to_predict=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_PREDICT'],
                                                                                max_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                min_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MIN_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                weeks_to_wait_after_the_upgrade=
                                                                                conf['TRAFFIC_IMPROVEMENT'][
                                                                                    'WEEKS_TO_WAIT_AFTER_UPGRADE'])

    
    file_name_voice = output_linear_regression_voice.path
    with open(file_name_voice, "rb") as file:
        model = pickle.load(file)        
    model_lr = model["model"]

    voix = predict_improvement_traffic_trend_kpis(df_increase_traffic_after_upgrade_voice,
                                                                                df_sites,
                                                                                model_lr,
                                                                                variable_to_group_by=['site_region'],
                                                                                max_yearly_increment=50,
                                                                                kpi_to_compute_trend=[
                                                                                    'total_voice_traffic_kerlands'],
                                                                                max_weeks_to_predict=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_PREDICT'],
                                                                                max_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MAX_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                min_weeks_to_consider_increase=
                                                                                conf['TRAFFIC_IMPROVEMENT_TREND'][
                                                                                    'MIN_WEEKS_TO_CONSIDER_INCREASE'],
                                                                                weeks_to_wait_after_the_upgrade=
                                                                                conf['TRAFFIC_IMPROVEMENT'][
                                                                                    'WEEKS_TO_WAIT_AFTER_UPGRADE'])

    df_predicted_increase_in_traffic_by_the_upgrade = merge_predicted_improvement_traffics(data_dl, data_ul, voix)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #artefacts pipelines
    df_traffic_features_future_upgrades_data_dl.to_csv(dataset_traffic_features_future_upgrades_data_dl.path, index=False)
    df_traffic_features_future_upgrades_data_ul.to_csv(dataset_traffic_features_future_upgrades_data_ul.path, index=False)
    df_traffic_features_future_upgrades_voice.to_csv(dataset_traffic_features_future_upgrades_voice.path, index=False)     
    df_traffic_features_future_upgrades_prediction_data_dl.to_csv(dataset_traffic_features_future_upgrades_prediction_data_dl.path, index=False)
    df_traffic_features_future_upgrades_prediction_data_ul.to_csv(dataset_traffic_features_future_upgrades_prediction_data_ul.path, index=False)
    df_traffic_features_future_upgrades_prediction_voice.to_csv(dataset_traffic_features_future_upgrades_prediction_voice.path, index=False)
    
    df_increase_traffic_after_upgrade_data_dl.to_csv(dataset_increase_traffic_after_upgrade_data_dl.path, index=False)
    df_increase_traffic_after_upgrade_data_ul.to_csv(dataset_increase_traffic_after_upgrade_data_ul.path, index=False)
    df_increase_traffic_after_upgrade_voice.to_csv(dataset_increase_traffic_after_upgrade_voice.path, index=False)
    df_predicted_increase_in_traffic_by_the_upgrade.to_csv(dataset_predicted_increase_in_traffic_by_the_upgrade.path, index=False)     
    
    
    
    
# USE TIMESTAMP TO DEFINE UNIQUE PIPELINE NAMES
#TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
#DISPLAY_NAME = 'pipeline-smartcapex-job{}'.format(TIMESTAMP)

@component(
base_image=BASE_IMAGE,
)
def compute_increase_of_arpu_by_the_upgrade (
    dataset_predicted_increase_in_traffic_by_the_upgrade: Input[Dataset],
    dataset_unit_prices: Input[Dataset],
    dataset_increase_of_arpu_by_the_upgrade: Output[Dataset],
    
):
    #imports
    from src.d00_conf.conf import conf, conf_loader
    import pandas as pd
    import datetime
        
    conf_loader("OSN")
    
       
    df_predicted_increase_in_traffic_by_the_upgrade = pd.read_csv(dataset_predicted_increase_in_traffic_by_the_upgrade.path)
    df_unit_prices_voice_and_data = pd.read_csv(dataset_unit_prices.path)
    
    # Renommage des colonnes pour correspondre aux noms souhaités
    df_unit_prices_voice_and_data.rename(columns={'ppm_data': 'unit_price_data_go'
                                                  #'ppm_voix': 'unit_price_voice_min'
                                                  },
                                         inplace=True)
    
    # Création de colonnes supplémentaires pour prendre en compte la diminution des prix
    df_unit_prices_voice_and_data[
        'unit_price_data_with_the_decrease'] = df_unit_prices_voice_and_data.unit_price_data_go
    
    # df_unit_prices_voice_and_data[
    #     'unit_price_voice_with_the_decrease'] = df_unit_prices_voice_and_data.unit_price_voice_min
    
    

    
    # Fusion des données d'augmentation du trafic prédites et des prix unitaires
    df_increase_of_arpu_by_the_upgrade = df_predicted_increase_in_traffic_by_the_upgrade.merge(df_unit_prices_voice_and_data,
                                                                                on='year', how='left')
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # iterating the columns
    for col in df_increase_of_arpu_by_the_upgrade.columns:
        print(col)

    # Calcul de l'augmentation de l'ARPU due à la mise à niveau pour les données et la voix
    df_increase_of_arpu_by_the_upgrade['arpu_increase_due_to_the_upgrade_data_dl_xof'] = \
        df_increase_of_arpu_by_the_upgrade.unit_price_data_with_the_decrease * \
        df_increase_of_arpu_by_the_upgrade.traffic_increase_due_to_the_upgrade_data#_dl

    df_increase_of_arpu_by_the_upgrade['arpu_increase_due_to_the_upgrade_data_ul_xof'] = \
        df_increase_of_arpu_by_the_upgrade.unit_price_data_with_the_decrease * \
        df_increase_of_arpu_by_the_upgrade.traffic_increase_due_to_the_upgrade_data#_ul

    # df_increase_of_arpu_by_the_upgrade['arpu_increase_due_to_the_upgrade_voice_xof'] = \
    #     df_increase_of_arpu_by_the_upgrade.unit_price_voice_with_the_decrease * \
    #     df_increase_of_arpu_by_the_upgrade.traffic_increase_due_to_the_upgrade_voice


    df_increase_of_arpu_by_the_upgrade['arpu_increase_due_to_the_upgrade_data_xof'] = df_increase_of_arpu_by_the_upgrade \
        ['arpu_increase_due_to_the_upgrade_data_dl_xof'] + df_increase_of_arpu_by_the_upgrade['arpu_increase_due_to_the_upgrade_data_ul_xof']
    
    df_increase_of_arpu_by_the_upgrade.to_csv(dataset_increase_of_arpu_by_the_upgrade.path, index=False)



#F2    
    
@component(
base_image=BASE_IMAGE,
)
def compute_revenu_per_traffic_unit (
    dataset_oss_preprocessed: Input[Dataset],
    dataset_unit_prices: Input[Dataset],
    dataset_charge_total_opex: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    dataset_aupu: Input[Dataset],
    dataset_site_revenue: Output[Dataset]
    
):
    
    #imports
    from src.d00_conf.conf import conf, conf_loader
    import pandas as pd
    import datetime
        
    conf_loader("OSN")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.mode.chained_assignment = None 

    df_oss_counter = pd.read_csv(dataset_oss_preprocessed.path)
    df_unit_price = pd.read_csv(dataset_unit_prices.path)
    df_charge_total_opex = pd.read_csv(dataset_charge_total_opex.path)
    df_sites = pd.read_csv(dataset_site_preprocessed.path)
    df_aupu = pd.read_csv(dataset_aupu.path)
    
    
    df_oss_counter = df_oss_counter[df_oss_counter.cell_tech.isin(['4G'])]
    
    df_unit_price["year"] = df_unit_price["year"].astype(str)
    # Conversion de la colonne 'date' en format datetime
    df_oss_counter['date'] = df_oss_counter['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d"))
    
    # Extraction des années et des mois à partir de la colonne 'date'
    df_oss_counter[['year', 'month']] = df_oss_counter['date'].apply(
        lambda x: pd.Series([str(x.year), str(x.month).zfill(2)]))
    df_oss_counter['period'] = df_oss_counter['year'] + df_oss_counter['month']

    # Regroupement des données par site, période, année et mois, et calcul des indicateurs de trafic
    df_service_agg = df_oss_counter.groupby(['site_id', 'period', 'year', 'month']).agg({
        'total_data_traffic_dl_gb': sum,
        'total_data_traffic_ul_gb': sum
        #'total_voice_traffic_kerlands': sum
    }).reset_index()
    
    # Fusion avec les prix unitaires en fonction de l'année
    df_service_agg = df_service_agg.merge(df_unit_price, on='year', how='left')

    # Calcul des revenus pour le trafic de données et de voix
    df_service_agg['revenues_dl_data_traffic'] = df_service_agg['total_data_traffic_dl_gb'] * df_service_agg['ppm_data']
    df_service_agg['revenues_ul_data_traffic'] = df_service_agg['total_data_traffic_ul_gb'] * df_service_agg['ppm_data']
    df_service_agg['revenues_data_traffic'] = df_service_agg['revenues_dl_data_traffic'] +  df_service_agg['revenues_ul_data_traffic']
   # df_service_agg['revenues_voice_traffic'] = df_service_agg['total_voice_traffic_kerlands'] * df_service_agg[
    #    'ppm_voix']
    #df_service_agg['revenue_total'] = df_service_agg['revenues_data_traffic'] + df_service_agg['revenues_voice_traffic']
    df_service_agg['revenue_total'] = df_service_agg['revenues_data_traffic']
    # Renommage des colonnes pour correspondre aux noms souhaités
    df_service_agg.rename(columns={'total_data_traffic_dl_gb': 'traffic_dl_kpis_data',
                                   'total_data_traffic_ul_gb': 'traffic_ul_kpis_data',
                                   #'total_voice_traffic_kerlands': 'traffic_kpis_voice',
                                   'ppm_data': 'unit_price_data',
                                   #'ppm_voix': 'unit_price_voice_min'
                                   },
                          inplace=True)

    # Création d'un dataframe pour les revenus par site
    df_site_revenue = df_service_agg[['site_id', 'period', 'traffic_dl_kpis_data','traffic_ul_kpis_data'
                                      , 'unit_price_data', 'revenues_data_traffic', 'revenue_total']] # 'revenues_voice_traffic', 'traffic_kpis_voice','unit_price_voice_min'
    
    df_site_revenue['charge_opex'] = df_charge_total_opex.sum(axis=1)[0] 
    df_site_revenue = df_site_revenue.merge(df_sites[["site_id", "site_region"]].drop_duplicates(), on='site_id', how='left')
    
    df_aupu.rename(columns={'Region': 'site_region'}, inplace=True)

    df_site_revenue = df_site_revenue.merge(df_aupu, on='site_region', how='left')
    
    
    ##################### calcul des charges qui s'appliquent uniquement à la data( depuis SONATEL nous
    # a indiqué de restreintre le périmètre uniquement à la data 4G)
    df_site_revenue["charge_pub_et_promotion"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
         'PUB_ET_PROMOTION'] / 100  # 1, 7 %
    df_site_revenue["charge_cst"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
         'CST'] / 100  # 4, 50 %
    df_site_revenue["charge_pse"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
         'PSE'] / 100  # 0, 00 %

    ##################### calcul des charges qui s'applique au revenu total
    df_site_revenue["charge_service_client"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'SERVICE_CLIENT'] / 100  # 1 %
    df_site_revenue["charge_msa"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'MSA'] / 100  # 15, 13 %
    df_site_revenue["charge_fghmf"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'FGHMF']  / 100  # 3,5 %
    df_site_revenue["charge_mf"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'MF'] / 100  # 1,4 %
    df_site_revenue["charge_patente"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'PATENTE'] / 100  # 0,4 %
    df_site_revenue["charge_cd"] = df_site_revenue["revenues_data_traffic"] * conf['OPEX_GESTION'][
        'CD'] / 100  # 6,0 %


    df_site_revenue["total_opex_site"] = df_site_revenue["charge_pub_et_promotion"] + \
                                         df_site_revenue["charge_cst"] + \
                                         df_site_revenue["charge_pse"] + \
                                         df_site_revenue["charge_service_client"] + \
                                         df_site_revenue["charge_msa"] + df_site_revenue["charge_fghmf"] + \
                                         df_site_revenue["charge_mf"] + df_site_revenue["charge_patente"] + \
                                         df_site_revenue["charge_cd"]

    # Calcul des charges pour les revenus de voix et de data par sites
    # df_site_revenue["revenues_total_traffic"] = df_site_revenue["revenues_voice_traffic"] + df_site_revenue[
    #    "revenues_data_traffic"]
    df_site_revenue["revenues_total_traffic"] =   df_site_revenue["revenues_data_traffic"]

    # Calcul des charges totales par site

    # Renommage des colonnes pour les revenus de voix et de data par site
    df_site_revenue["REVENU_DATA_SITE"] = df_site_revenue['revenues_data_traffic']
    # df_site_revenue["REVENU_VOIX_SITE"] = df_site_revenue['revenues_voice_traffic']
    df_site_revenue = df_site_revenue.drop_duplicates()
    
    # Enregistrement du dataframe des revenus par site
    
    df_site_revenue.to_csv(dataset_site_revenue.path, index=False)
    
    
  
    
    
    
@component(
base_image=BASE_IMAGE,
)
def compute_site_margin (
    dataset_site_revenue: Input[Dataset],
    dataset_margin_per_site: Output[Dataset]
    
    
):
    
    from src.d00_conf.conf import conf, conf_loader
    import pandas as pd
    import datetime
        
    conf_loader("OSN")
    
    revenues_per_unit_traffic = pd.read_csv(dataset_site_revenue.path)
    
    # Calcul de la marge mensuelle par site pour les revenus de données et de voix
    revenues_per_unit_traffic['site_margin_monthly'] = revenues_per_unit_traffic['revenues_total_traffic'] - \
                                                            revenues_per_unit_traffic['total_opex_site']


    # Sélection des 12 derniers mois
    last_periods = list(revenues_per_unit_traffic.period.unique())
    last_periods.sort(reverse=True)
    last_periods = last_periods[:12]
    revenues_per_unit_traffic = revenues_per_unit_traffic[revenues_per_unit_traffic.period.isin(last_periods)].copy()

    # Calcul de la marge annuelle par site pour les revenus de données et de voix
    df_margin_per_site = revenues_per_unit_traffic.groupby(['site_id'])[['site_margin_monthly'
                                                                         ]].sum().reset_index()
    df_margin_per_site.columns = ['site_id', 'site_margin_yearly']

    
    df_margin_per_site.to_csv(dataset_margin_per_site.path, index=False)
    
    

    
@component(
base_image=BASE_IMAGE,
)    
def compute_increase_of_yearly_site_margin(
     dataset_site_revenue: Input[Dataset],
     dataset_increase_of_arpu_by_the_upgrade: Input[Dataset],
     dataset_margin_per_site: Input[Dataset],
     dataset_increase_arpu_by_year: Output[Dataset] ):
        
    from src.d00_conf.conf import conf, conf_loader
    from src.d01_utils.utils import get_month_year_period
    import pandas as pd
    import datetime
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #pd.options.mode.chained_assignment = None 
    
    conf_loader("OSN")
    
    
    
    revenues_per_unit_traffic = pd.read_csv(dataset_site_revenue.path)
    df_increase_arpu_due_to_the_upgrade = pd.read_csv(dataset_increase_of_arpu_by_the_upgrade.path)
    df_margin_per_site = pd.read_csv(dataset_margin_per_site.path)
    
    
    
    
    
    # Grouper les données d'augmentation d'ARPU par site, groupe de bandes et année, et calculer la somme des augmentations d'ARPU
    df_increase_arpu_by_year = df_increase_arpu_due_to_the_upgrade.groupby(['site_id', 'bands_upgraded', 'year'])[
        ['arpu_increase_due_to_the_upgrade_data_xof']].sum().reset_index() #'arpu_increase_due_to_the_upgrade_voice_xof'

    # Convertir la colonne de date en format datetime
    df_increase_arpu_due_to_the_upgrade['date'] = pd.to_datetime(df_increase_arpu_due_to_the_upgrade['date'])

    # Extraire le mois et l'année à partir de la date et calculer le mois périodique
    df_increase_arpu_due_to_the_upgrade['month_period'] = df_increase_arpu_due_to_the_upgrade['date'].apply(
        get_month_year_period)

    # Compter le nombre de mois par site et par année
    df_months_per_year = \
    df_increase_arpu_due_to_the_upgrade[['site_id', 'year', 'month_period']].drop_duplicates().groupby(
        ['site_id', 'year'])['month_period'].count().reset_index()
    df_months_per_year.columns = ['site_id', 'year', 'number_of_months_per_year']

    # Fusionner les données d'augmentation d'ARPU par année avec le nombre de mois par année
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_months_per_year,
                                                              on=['site_id', 'year'],
                                                              how='left')

    # Sélectionner les 12 dernières périodes de revenus par unité de trafic
    last_periods = list(revenues_per_unit_traffic.period.unique())
    last_periods.sort(reverse=True)
    last_periods = last_periods[:12]
    revenues_per_unit_traffic = revenues_per_unit_traffic[revenues_per_unit_traffic.period.isin(last_periods)].copy()

    # Calculer les revenus annuels par service pour chaque site
    df_annual_revenues_by_service = revenues_per_unit_traffic.groupby(['site_id'])[['REVENU_DATA_SITE']].sum().reset_index() #'REVENU_VOIX_SITE'
    df_annual_revenues_by_service.columns = ['site_id', 'annual_revenues_data_traffic'] # 'annual_revenues_voice_traffic'

    # Fusionner les données d'augmentation d'ARPU par année avec les revenus annuels par service
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_annual_revenues_by_service,
                                                              on='site_id',
                                                              how='left')

    # Calculer les augmentations annuelles de revenus pour la data, la voix et l'ensemble
    df_increase_arpu_by_year['increase_yearly_data_revenues'] = df_increase_arpu_by_year[
                                                                    'arpu_increase_due_to_the_upgrade_data_xof'] / \
                                                                df_increase_arpu_by_year['annual_revenues_data_traffic']
    # df_increase_arpu_by_year['increase_yearly_voice_revenues'] = df_increase_arpu_by_year[
    #                                                                  'arpu_increase_due_to_the_upgrade_voice_xof'] / \
    #                                                              df_increase_arpu_by_year[
    #                                                                  'annual_revenues_voice_traffic']
    df_increase_arpu_by_year['increase_yearly_revenues'] = df_increase_arpu_by_year['arpu_increase_due_to_the_upgrade_data_xof'] / df_increase_arpu_by_year['annual_revenues_data_traffic']


    # Fusionner les données d'augmentation d'ARPU par année avec la marge annuelle par site
    df_increase_arpu_by_year = df_increase_arpu_by_year.merge(df_margin_per_site[['site_id', 'site_margin_yearly']],
                                                              on='site_id',
                                                              how='left')

    # Calculer l'augmentation de marge annuelle due à la mise à niveau
    df_increase_arpu_by_year['increase_yearly_margin_due_to_the_upgrade'] = df_increase_arpu_by_year[
                                                                                'increase_yearly_revenues'] * \
                                                                            df_increase_arpu_by_year[
                                                                                'site_margin_yearly']

    df_increase_arpu_by_year.to_csv(dataset_increase_arpu_by_year.path, index=False)
    
    

 
    
@component(
base_image=BASE_IMAGE,
)
def compute_increase_cash_flow (
    dataset_increase_arpu_by_year: Input[Dataset],
    dataset_site_preprocessed: Input[Dataset],
    dataset_increase_of_arpu_by_the_upgrade: Input[Dataset],
    dataset_selected_band_per_site: Input[Dataset],
    dataset_final: Output[Dataset]

):
    
    from src.d00_conf.conf import conf, conf_loader
    from src.d01_utils.utils import get_month_year_period
    import pandas as pd
    import datetime
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.mode.chained_assignment = None
        
    conf_loader("OSN")
    
    df_capex =  pd.read_csv(dataset_increase_arpu_by_year.path) #dataset_xxx to be defined 
    
    
    df_sites =   pd.read_csv(dataset_site_preprocessed.path)
    df_increase_in_margin_due_to_the_upgrade =  pd.read_csv(dataset_increase_of_arpu_by_the_upgrade.path)
    
    
    
    # iterating the columns
    print("1111111111111111111111111111111111111111111111111111111")
    for col in df_increase_in_margin_due_to_the_upgrade.columns:
        print(col)
    
    selected_band_per_site =  pd.read_csv(dataset_selected_band_per_site.path)
    

    # Remplacement des valeurs dans df_increase_in_margin_due_to_the_upgrade
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.replace({'bands_upgraded': {
        'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900', 'F1_U2100': 'U2100', 'F4_U2100': 'U2100',
        'F2_U900': 'U900', 'split1800': 'Split1800', 'split2600': 'Split2600'}})

    # Renommage de la colonne 'opex' en 'opex_costs' dans df_opex
    df_opex = df_sites[['site_id']].drop_duplicates()
    df_opex['opex'] =conf['NPV']['OPEX_THIES']
    df_opex = df_opex.rename(columns={"opex": "opex_costs"})
    print(" opex costs year ")
    print(df_opex.head())
    # Calcul de l'année minimale par site dans df_increase_in_margin_due_to_the_upgrade
    df_min_year = df_increase_in_margin_due_to_the_upgrade.groupby(['site_id'])['year'].min().reset_index()
    df_min_year.columns = ['site_id', 'min_year']
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(df_min_year, on='site_id',
                                                                                              how='left')
    # Calcul de l'année de cash flow relative dans df_increase_in_margin_due_to_the_upgrade
    df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = (df_increase_in_margin_due_to_the_upgrade[
                                                                      'year'].apply(int) -
                                                                  df_increase_in_margin_due_to_the_upgrade[
                                                                      'min_year'].apply(int)) + 1

    #df_increase_in_margin_due_to_the_upgrade['cash_flow_year'] = df_increase_in_margin_due_to_the_upgrade[
    #    'cash_flow_year'].astype(int)
    
    
    # Fusion avec df_sites pour obtenir les informations du site
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(
        df_sites[["site_id", "site_constructor"]].drop_duplicates(), on=['site_id'], how='left')

    # Renommage de la colonne 'band_number' en 'nb_sectors' dans selected_band_per_site
    selected_band_per_site = selected_band_per_site.rename(columns={'band_number': 'nb_sectors'})
    # Remplacement de valeurs dans selected_band_per_site
    selected_band_per_site = selected_band_per_site.replace({'bands_upgraded': {'F2_U2100': 'U2100',
                                                                                'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                                                'F1_U2100': 'U2100',
                                                                                'F4_U2100': 'U2100', 'F2_U900': 'U900',
                                                                                'split1800': 'Split1800',
                                                                                'split2600': 'Split2600'}})

    selected_band_per_site.drop_duplicates(subset=['site_id', 'bands_upgraded'], inplace=True)

    # Fusion avec selected_band_per_site pour obtenir les informations sur les bandes et le nombre de secteurs
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(
        selected_band_per_site[['site_id', 'bands_upgraded', 'nb_sectors']], on=['site_id', 'bands_upgraded'],
        how='left')
    # Renommage de la colonne 'bands_upgraded' en 'cell_band'
    df_increase_in_margin_due_to_the_upgrade.rename(columns={'bands_upgraded': 'cell_band'}, inplace=True)

    # Fusion avec df_opex pour obtenir les coûts d'opex
    df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.merge(df_opex, on=['site_id'],
                                                                                              how='left')

    # Calcul du coût d'opex annuel

    ####################################### to do : check des opex sites , mensuel ou annuel ######################################
    # df_increase_in_margin_due_to_the_upgrade['opex_cost_year'] = df_increase_in_margin_due_to_the_upgrade[
    #                                                                  'opex_costs'] * \
    #                                                              df_increase_in_margin_due_to_the_upgrade[
    #                                                                  'number_of_months_per_year']

    df_increase_in_margin_due_to_the_upgrade['opex_cost_year'] = df_increase_in_margin_due_to_the_upgrade[
        'opex_costs']

    # Renommage de la colonne 'increase_yearly_total_margin_due_to_the_upgrade' en 'increase_yearly_margin_due_to_the_upgrade'
    #df_increase_in_margin_due_to_the_upgrade = df_increase_in_margin_due_to_the_upgrade.rename(
    #    columns={"increase_yearly_total_margin_due_to_the_upgrade": "increase_yearly_margin_due_to_the_upgrade"})

    print(" df_increase_in_margin_due_to_the_upgrade costs year after cash flow")
    print(df_increase_in_margin_due_to_the_upgrade.head())
    # Calcul de l'augmentation de cash flow due à la mise à niveau
    df_increase_in_margin_due_to_the_upgrade['increase_cash_flow_due_to_the_upgrade'] = \
        df_increase_in_margin_due_to_the_upgrade['increase_yearly_margin_due_to_the_upgrade'] - \
        df_increase_in_margin_due_to_the_upgrade['opex_cost_year']

    print(" df_increase_in_margin_due_to_the_upgrade costs year after cash flow")


    df_capex = df_capex.replace({'bands_upgraded': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                    'F1_U2100': 'U2100',
                                                    'F4_U2100': 'U2100', 'F2_U900': 'U900',
                                                    'split1800': 'Split1800',
                                                    'split2600': 'Split2600'}})

    # Renommage de la colonne 'bands_upgraded' en 'cell_band' dans df_capex
    df_capex.rename(columns={'bands_upgraded': 'cell_band'}, inplace=True)

    # Ajout des colonnes 'cash_flow_year' et 'increase_cash_flow_due_to_the_upgrade' dans df_capex
    df_capex['cash_flow_year'] = 0
    df_capex['increase_cash_flow_due_to_the_upgrade'] = -df_capex['capex']
    df_capex.drop(columns='capex', inplace=True)

    # Fusion avec df_capex pour obtenir les informations sur les sites, les bandes et le nombre de secteurs
    df_capex_site = df_increase_in_margin_due_to_the_upgrade[
        ['site_id', 'site_constructor', 'cell_band', 'nb_sectors']].drop_duplicates().merge(df_capex,
                                                                                            on=['site_constructor',
                                                                                                'cell_band',
                                                                                                'nb_sectors'],
                                                                                            how='left')
    # Conversion de la colonne 'increase_cash_flow_due_to_the_upgrade' en int64
    #df_increase_in_margin_due_to_the_upgrade['increase_cash_flow_due_to_the_upgrade'] = \
    #    df_increase_in_margin_due_to_the_upgrade['increase_cash_flow_due_to_the_upgrade'].astype('int64')

    # Concaténation de df_increase_in_margin_due_to_the_upgrade et df_capex_site
    df_final = pd.concat([df_increase_in_margin_due_to_the_upgrade, df_capex_site])
    df_final = df_final.reset_index(drop=True)

    # Tri par ordre croissant des années de cash flow
    df_final = df_final.sort_values('cash_flow_year', ascending=True)


    df_final.to_csv(dataset_final.path, index=False)


    
    
@component(
base_image=BASE_IMAGE,
)
def compute_npv (
    dataset_final: Input[Dataset],
    dataset_increase_of_arpu_by_the_upgrade: Input[Dataset],
    dataset_final_npv_of_the_upgrade: Output[Dataset]

):    
    
        
    from src.d00_conf.conf import conf, conf_loader
    from src.d01_utils.utils import npv_since_2nd_years, calculate_npv, irr, compute_npv_aux
    import pandas as pd
    import numpy as np
    import numpy_financial as npf
    import datetime
    conf_loader("OSN")
    
    
#     def compute_npv_aux(df, wacc):
#         # Fonction auxiliaire pour calculer le NPV
#         df = df.reset_index()
#         df = df[['cash_flow_year', 'increase_cash_flow_due_to_the_upgrade']].sort_values(by='cash_flow_year',
#                                                                                          ascending=True)
#         return npv_since_2nd_years(values=df['increase_cash_flow_due_to_the_upgrade'].values, rate=wacc / 100)

    wacc=conf['NPV']['WACC']
    
    df_cash_flow = pd.read_csv(dataset_final.path)
    df_arpu = pd.read_csv(dataset_increase_of_arpu_by_the_upgrade.path)
    
    
    nb_year = df_cash_flow['cash_flow_year'].unique()
    nb_year = np.sort(nb_year)

    # Calcul des flux de trésorerie actualisés sans escompte
    df_cash_flow_discount = df_cash_flow[~(df_cash_flow['cash_flow_year'].isin([nb_year[0]]))]
    df_cash_flow_no_discount = df_cash_flow[df_cash_flow['cash_flow_year'].isin([nb_year[0]])]

    df_cash_flow_no_discount_npv = df_cash_flow_no_discount.groupby(['site_id', 'cell_band'])[
        ['increase_cash_flow_due_to_the_upgrade']].sum().reset_index()
    df_cash_flow_no_discount_npv.columns = ['site_id', 'cell_band', 'capex_cf_y1']

    # df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
    #     lambda x: compute_npv_aux(x, wacc)).reset_index()
    
    # df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
    #     lambda x: x.reset_index()[['cash_flow_year', 'increase_cash_flow_due_to_the_upgrade']]
    #     .sort_values(by='cash_flow_year', ascending=True)['increase_cash_flow_due_to_the_upgrade']
    #     .apply(lambda values: npv_since_2nd_years(values.values, wacc / 100))
    #     .sum()).reset_index()
    
    #df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
    #lambda x: npv_since_2nd_years(x['increase_cash_flow_due_to_the_upgrade'].values, wacc / 100)).reset_index()
 
    
    #df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
    #lambda x: npv_since_2nd_years(wacc / 100, x['increase_cash_flow_due_to_the_upgrade'].values)).reset_index()
    
    # df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
    # lambda x: npv_since_2nd_years(wacc / 100, x['increase_cash_flow_due_to_the_upgrade'].values.tolist())).reset_index()
    
    df_cash_flow_discount_npv = df_cash_flow_discount.groupby(['site_id', 'cell_band']).apply(
        lambda x: compute_npv_aux(x, wacc)).reset_index()
    
    
    df_cash_flow_discount_npv.columns = ['site_id', 'cell_band', 'NPV_cf_y2']

    df_npv = df_cash_flow_discount_npv.merge(df_cash_flow_no_discount_npv, on=['site_id', 'cell_band'], how='left')

    df_npv['NPV'] = df_npv['capex_cf_y1'] + df_npv['NPV_cf_y2']

    df_npv.drop(['capex_cf_y1', 'NPV_cf_y2'], axis=1, inplace=True)

    df_npv.columns = ['site_id', 'cell_band', 'NPV']

    # Pivoter les flux de trésorerie par année

    df_cash_flow_pv = pd.pivot_table(df_cash_flow, values='increase_cash_flow_due_to_the_upgrade',
                                     index=['site_id', 'cell_band'], columns=['cash_flow_year'], aggfunc=np.sum)

    new_columns_names = []
    for i in df_cash_flow_pv.columns:
        new_columns_names.append('cash_flow_year_' + str(int(i)))
    df_cash_flow_pv.columns = new_columns_names
    df_cash_flow_pv.reset_index(inplace=True)

    # Pivoter les coûts opérationnels par année
    df_opex_cost_year_pv = pd.pivot_table(df_cash_flow, values='opex_costs', index=['site_id', 'cell_band'],
                                          columns=['cash_flow_year'], aggfunc=np.sum)

    opex_columns_names = []
    for i in df_opex_cost_year_pv.columns:
        opex_columns_names.append('opex_cost_year_' + str(i))
    df_opex_cost_year_pv.columns = opex_columns_names
    df_opex_cost_year_pv.reset_index(inplace=True)

    df = df_cash_flow[['site_id', 'cell_band']].drop_duplicates()

    # Fusionner les données des flux de trésorerie et des coûts opérationnels
    df = df.merge(df_cash_flow_pv, on=['site_id', 'cell_band'], how='left')
    df = df.merge(df_opex_cost_year_pv, on=['site_id', 'cell_band'], how='left')

    # Fusionner avec les NPV
    df = df.merge(df_npv, on=['site_id', 'cell_band'], how='left')

    #df_arpu = pd.read_csv(os.path.join(conf['PATH']['PROCESSED_DATA'], "df_increase_of_arpu_by_the_upgrade.csv"),
    

    
    df_increase_arpu_due_to_the_upgrade = df_arpu.groupby(['site_id', 'bands_upgraded'])[
        ['arpu_increase_due_to_the_upgrade_data_xof']].sum().reset_index()
    df_increase_arpu_due_to_the_upgrade['increase_arpu_due_to_the_upgrade'] = df_increase_arpu_due_to_the_upgrade[
        'arpu_increase_due_to_the_upgrade_data_xof']
    df_increase_arpu_due_to_the_upgrade = df_increase_arpu_due_to_the_upgrade[
        ['site_id', 'bands_upgraded', 'increase_arpu_due_to_the_upgrade']]
    df_increase_arpu_due_to_the_upgrade.rename(
        columns={'increase_arpu_due_to_the_upgrade': 'total_revenue', 'bands_upgraded': 'cell_band'}, inplace=True)

    df = df.merge(df_increase_arpu_due_to_the_upgrade, on=['site_id', 'cell_band'], how='left')

    df['total_opex'] = df[[col for col in df.columns if col.startswith('opex')]].sum(axis=1)

    df['EBITDA_Value'] = df['total_revenue'] - df['total_opex']
    df['EBITDA'] = df['EBITDA_Value'] / df['total_revenue']

    # Calculer le taux de rendement interne (IRR)
    df_irr_columns = ['site_id', 'cell_band']
    df_irr_columns = df_irr_columns + new_columns_names
    df_irr_0_1 = df[df_irr_columns]
    df_irr_0_1 = df_irr_0_1.fillna(0)
    df_irr_0_1['cash_flow_years_0_1'] = df_irr_0_1['cash_flow_year_0'].astype(float) + df_irr_0_1[
        'cash_flow_year_1'].astype(float)
    df_irr_0_1.drop(['cash_flow_year_0', 'cash_flow_year_1'], axis=1, inplace=True)
    df_irr_0_1 = df_irr_0_1.reindex(columns=['site_id', 'cell_band', 'cash_flow_years_0_1'] + df_irr_columns[4:])
    irr_columns = df_irr_0_1.columns[2:]
    df_irr_0_1 = df_irr_0_1[df_irr_0_1.cash_flow_years_0_1.notna()]
    df_irr_0_1['IRR'] = df_irr_0_1[irr_columns].apply(lambda row: irr(row), axis=1)
    df_irr = df_irr_0_1[['site_id', 'cell_band', 'IRR']]

    df = df.merge(df_irr, on=['site_id', 'cell_band'], how='left')
    df.IRR = df.IRR.fillna(0)

    ##################

    df = calculate_npv(df, wacc)
    df.drop(columns='NPV', inplace=True)
    df.rename(columns={'cash_flow_discount_0': 'cash_flow_0',
                       'cash_flow_discount_1': 'cash_flow_1',
                       'cash_flow_discount_2': 'cash_flow_2',
                       'cash_flow_discount_3': 'cash_flow_3',
                       'cash_flow_discount_4': 'cash_flow_4',
                       'cash_flow_discount_5': 'cash_flow_5',
                       'cash_flow_discount_6': 'cash_flow_6',
                       'cash_flow_discount_7': 'cash_flow_7',
                       'cash_flow_discount_8': 'cash_flow_8',
                       'cash_flow_discount_9': 'cash_flow_9',
                       'cash_flow_discount_10': 'cash_flow_10',
                       'NPV_DISCOUNT': 'NPV'}, inplace=True)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # df.info
    # print(df)
    # iterating the columns
    for col in df.columns:
        print(col)
    # df_final_npv_of_the_upgrade = df[['site_id', 'cell_band', 'cash_flow_year_0', 'cash_flow_year_1',
    #          'cash_flow_year_2', 'cash_flow_year_3', 'cash_flow_year_4',
    #          'cash_flow_year_5', 'cash_flow_year_6', 'cash_flow_year_7',
    #          'cash_flow_year_8', 'cash_flow_year_9', 'cash_flow_year_10',
    #          'opex_cost_year_0', 'opex_cost_year_1', 'opex_cost_year_2',
    #          'opex_cost_year_3', 'opex_cost_year_4', 'opex_cost_year_5',
    #          'opex_cost_year_6', 'opex_cost_year_7', 'opex_cost_year_8',
    #          'opex_cost_year_9', 'opex_cost_year_10', 'NPV', 'total_revenue',
    #          'total_opex', 'EBITDA_Value', 'EBITDA', 'IRR']]
    
    df_final_npv_of_the_upgrade = df[['site_id', 'cell_band', 'cash_flow_year_0', 'cash_flow_year_1',
             'cash_flow_year_2', 'cash_flow_year_3', 'cash_flow_year_4',
             'cash_flow_year_5',    
             'opex_cost_year_0', 'opex_cost_year_1', 'opex_cost_year_2',
             'opex_cost_year_3', 'opex_cost_year_4', 'opex_cost_year_5',
             'NPV', 'total_revenue',
             'total_opex', 'EBITDA_Value', 'EBITDA', 'IRR']]
    
    df_final_npv_of_the_upgrade.to_csv(dataset_final_npv_of_the_upgrade.path, index=False)





    
# USE TIMESTAMP TO DEFINE UNIQUE PIPELINE NAMES
#TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
#DISPLAY_NAME = 'pipeline-smartcapex-job{}'.format(TIMESTAMP)

@dsl.pipeline(
     pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="smartcapex-pipeline", 
    
)

def pipeline(
     
    dataset_src: str,
    table_site_raw: str,   
    table_oss_2g_raw: str, 
    table_oss_3g_raw: str, 
    table_oss_4g_raw: str, 
    table_oss_rtx_raw: str,
    table_typology_sector_raw: str,
    table_unit_prices: str,
    table_charge_total_opex: str,
    table_aupu: str,
    table_capex: str,
    
    
    dataset_preprocessing: str,
    df_sites: str,
    oss_counter_weekly: str,
    
    dataset_intermediate : str,
    df_traffic_predicted :str,
    df_distance :str,
    df_selected_band_per_site :str,
    df_affected_cells :str,
    df_future_upgrades :str,
    df_cell_affected :str,
    list_of_upgrades :str,
    sites_to_remove :str,
    df_data_dl_traffic_features :str,
    df_data_ul_traffic_features :str,
    df_voice_traffic_features :str,
    df_traffic_by_region :str,
    
    df_traffic_features_future_upgrades_data_dl: str,
    df_traffic_features_future_upgrades_data_ul: str,
    df_traffic_features_future_upgrades_voice: str,
    df_traffic_features_future_upgrades_prediction_data_dl: str,
    df_traffic_features_future_upgrades_prediction_data_ul: str,
    df_traffic_features_future_upgrades_prediction_voice: str,
    df_increase_traffic_after_upgrade_data_dl: str,
    df_increase_traffic_after_upgrade_data_ul: str,
    df_increase_traffic_after_upgrade_voice: str,
    df_predicted_increase_in_traffic_by_the_upgrade: str,
    
    
    df_increase_of_arpu_by_the_upgrade: str,
    df_site_revenue: str,
    df_margin_per_site: str,
    df_increase_arpu_by_year: str,
    df_final: str,
    df_final_npv_of_the_upgrade: str,
    
    
    project: str = PROJECT_ID,
):

    get_data_sites_op = get_data(project,dataset_src,table_site_raw)
    get_data_oss_2g_op = get_data(project,dataset_src,table_oss_2g_raw)
    get_data_oss_3g_op = get_data(project,dataset_src,table_oss_3g_raw)
    get_data_oss_4g_op = get_data(project,dataset_src,table_oss_4g_raw)
    get_data_oss_rtx_op = get_data(project,dataset_src,table_oss_rtx_raw)
    get_data_typology_op = get_data(project,dataset_src,table_typology_sector_raw)

    get_unit_prices_op = get_data(project,dataset_src,table_unit_prices)
    
    get_charge_total_opex_op = get_data(project,dataset_src,table_charge_total_opex)
    
    get_aupu_op = get_data(project,dataset_src,table_aupu)
    get_capex_op = get_data(project,dataset_src,table_capex)
    
    
    data_preprocess_op = preprocess_data(get_data_sites_op.outputs["dataset_raw"],get_data_oss_2g_op.outputs["dataset_raw"],get_data_oss_3g_op.outputs["dataset_raw"],\
                                         get_data_oss_4g_op.outputs["dataset_raw"],get_data_oss_rtx_op.outputs["dataset_raw"])
    
    save_to_bigquery_sites_op = save_to_bigquery(data_preprocess_op.outputs["dataset_sites_preprocessed"],project,dataset_preprocessing,df_sites)
    save_to_bigquery_oss_op = save_to_bigquery(data_preprocess_op.outputs["dataset_oss_preprocessed"],project,dataset_preprocessing,oss_counter_weekly)
    
    traffic_forcasting_op = (traffic_forcasting(data_preprocess_op.outputs["dataset_oss_preprocessed"],data_preprocess_op.outputs["dataset_sites_preprocessed"])).set_cpu_limit('16').set_memory_limit('60G')
    save_to_bigquery_traffic_predicted_op = save_to_bigquery(traffic_forcasting_op.outputs["df_traffic_predicted"],project,dataset_intermediate,df_traffic_predicted)
    
    compute_sites_distances_op = compute_sites_distances(data_preprocess_op.outputs["dataset_sites_preprocessed"])
    save_to_bigquery_sites_distances_op = save_to_bigquery(compute_sites_distances_op.outputs["df_distance"],project,dataset_intermediate,df_distance)

    upgrade_selection_op = process_bands_to_upgrade(compute_sites_distances_op.outputs["df_distance"],traffic_forcasting_op.outputs["df_traffic_predicted"],data_preprocess_op.outputs["dataset_sites_preprocessed"],get_data_typology_op.outputs["dataset_raw"])
    


    save_to_bigquery_selected_band_per_site_op = save_to_bigquery(upgrade_selection_op.outputs["dataset_selected_band_per_site"],project,dataset_intermediate,df_selected_band_per_site)
    save_to_bigquery_affected_cells_op = save_to_bigquery(upgrade_selection_op.outputs["dataset_affected_cells"],project,dataset_intermediate,df_affected_cells)
    save_to_bigquery_future_upgrades_op = save_to_bigquery(upgrade_selection_op.outputs["dataset_cluster_future_upgrades"],project,dataset_intermediate,df_future_upgrades)
    
    
    train_technical_pipeline_op = train_technical_pipeline(data_preprocess_op.outputs["dataset_oss_preprocessed"],data_preprocess_op.outputs["dataset_sites_preprocessed"],compute_sites_distances_op.outputs["df_distance"])
    
    save_to_bigquery_cell_affected_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_cell_affected"],project,dataset_intermediate,df_cell_affected)
    save_to_bigquery_list_of_upgrades_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_list_of_upgrades"],project,dataset_intermediate,list_of_upgrades)
    #save_to_bigquery_sites_to_remove_op = #save_to_bigquery(train_technical_pipeline_op.outputs["dataset_sites_to_remove"],project,dataset_intermediate,sites_to_remove)
    save_to_bigquery_data_dl_traffic_features_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_data_dl_traffic_features"],project,dataset_intermediate,df_data_dl_traffic_features)
    save_to_bigquery_data_ul_traffic_features_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_data_ul_traffic_features"],project,dataset_intermediate,df_data_ul_traffic_features)
    save_to_bigquery_voice_traffic_features_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_voice_traffic_features"],project,dataset_intermediate,df_voice_traffic_features)
    save_to_bigquery_traffic_by_region_op = save_to_bigquery(train_technical_pipeline_op.outputs["dataset_traffic_by_region"],project,dataset_intermediate,df_traffic_by_region)


    print("11111111111111111111111upgrade_selection_op1111111111111111111111111")
    print(upgrade_selection_op.outputs)
    print("2222222222222222upgrade_selection_op22222222222222222")
    #print(upgrade_selection_op.outputs["dataset_affected_cells"])
    process_traffic_improvement_op = process_traffic_improvement(traffic_forcasting_op.outputs["df_traffic_predicted"], upgrade_selection_op.outputs["dataset_affected_cells"], upgrade_selection_op.outputs["dataset_cluster_future_upgrades"], upgrade_selection_op.outputs["dataset_selected_band_per_site"], data_preprocess_op.outputs["dataset_sites_preprocessed"], train_technical_pipeline_op.outputs["output_data_dl_model"], train_technical_pipeline_op.outputs["output_data_ul_model"], train_technical_pipeline_op.outputs["output_voice_model"], train_technical_pipeline_op.outputs["output_linear_regression_data_dl"], train_technical_pipeline_op.outputs["output_linear_regression_data_ul"], train_technical_pipeline_op.outputs["output_linear_regression_voice"])
                                                                

    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_data_dl"],project,dataset_intermediate,df_traffic_features_future_upgrades_data_dl)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_data_ul"],project,dataset_intermediate,df_traffic_features_future_upgrades_data_ul)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_voice"],project,dataset_intermediate,df_traffic_features_future_upgrades_voice)   
    
    
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_prediction_data_dl"],project,dataset_intermediate,df_traffic_features_future_upgrades_prediction_data_dl)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_prediction_data_ul"],project,dataset_intermediate,df_traffic_features_future_upgrades_prediction_data_ul)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_traffic_features_future_upgrades_prediction_voice"],project,dataset_intermediate,df_traffic_features_future_upgrades_prediction_voice)
    
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_increase_traffic_after_upgrade_data_dl"],project,dataset_intermediate,df_increase_traffic_after_upgrade_data_dl)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_increase_traffic_after_upgrade_data_ul"],project,dataset_intermediate,df_increase_traffic_after_upgrade_data_ul)
    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_increase_traffic_after_upgrade_voice"],project,dataset_intermediate,df_increase_traffic_after_upgrade_voice )

    save_to_bigquery_process_traffic_improvement_op = save_to_bigquery(process_traffic_improvement_op.outputs["dataset_predicted_increase_in_traffic_by_the_upgrade"],project,dataset_intermediate,df_predicted_increase_in_traffic_by_the_upgrade)
    
    
    
    compute_increase_of_arpu_by_the_upgrade_op = compute_increase_of_arpu_by_the_upgrade(process_traffic_improvement_op.outputs["dataset_predicted_increase_in_traffic_by_the_upgrade"], get_unit_prices_op.outputs["dataset_raw"])
          
    save_to_bigquery_compute_increase_of_arpu_by_the_upgrade_op = save_to_bigquery(compute_increase_of_arpu_by_the_upgrade_op.outputs["dataset_increase_of_arpu_by_the_upgrade"],project,dataset_intermediate,df_increase_of_arpu_by_the_upgrade)
    
    
    compute_revenu_per_traffic_unit_op = compute_revenu_per_traffic_unit(data_preprocess_op.outputs["dataset_oss_preprocessed"], get_unit_prices_op.outputs["dataset_raw"], get_charge_total_opex_op.outputs["dataset_raw"], data_preprocess_op.outputs["dataset_sites_preprocessed"], get_aupu_op.outputs["dataset_raw"])
    
    save_to_bigquery_compute_revenu_per_traffic_unit_op = save_to_bigquery(compute_revenu_per_traffic_unit_op.outputs["dataset_site_revenue"],project,dataset_intermediate,df_site_revenue)
    
    
    
    compute_site_margin_op = compute_site_margin(compute_revenu_per_traffic_unit_op.outputs["dataset_site_revenue"])
    
    save_to_bigquery_compute_site_margin_op = save_to_bigquery(compute_site_margin_op.outputs["dataset_margin_per_site"],project,dataset_intermediate,df_margin_per_site)
 
    
    compute_increase_of_yearly_site_margin_op = compute_increase_of_yearly_site_margin(compute_revenu_per_traffic_unit_op.outputs["dataset_site_revenue"], compute_increase_of_arpu_by_the_upgrade_op.outputs["dataset_increase_of_arpu_by_the_upgrade"], compute_site_margin_op.outputs["dataset_margin_per_site"])
    
    
    save_to_bigquery_compute_increase_of_yearly_site_margin_op = save_to_bigquery(compute_increase_of_yearly_site_margin_op.outputs["dataset_increase_arpu_by_year"],project,dataset_intermediate,df_increase_arpu_by_year)


    
    compute_increase_cash_flow_op = compute_increase_cash_flow(get_capex_op.outputs["dataset_raw"], data_preprocess_op.outputs["dataset_sites_preprocessed"],
    compute_increase_of_yearly_site_margin_op.outputs["dataset_increase_arpu_by_year"], upgrade_selection_op.outputs["dataset_selected_band_per_site"])
    
    
    save_to_bigquery_compute_increase_cash_flow_op = save_to_bigquery(compute_increase_cash_flow_op.outputs["dataset_final"],project,dataset_intermediate,df_final)

    
    compute_npv_aux_op = compute_npv(compute_increase_cash_flow_op.outputs["dataset_final"], compute_increase_of_arpu_by_the_upgrade_op.outputs["dataset_increase_of_arpu_by_the_upgrade"])

    save_to_bigquery_compute_npv_aux_op = save_to_bigquery(compute_npv_aux_op.outputs["dataset_final_npv_of_the_upgrade"],project,dataset_intermediate,df_final_npv_of_the_upgrade)
    
 #####################fin composants#####################        
              
# compute_increase_of_arpu_by_the_upgrade  df_increase_of_arpu_by_the_upgrade

# compute_revenu_per_traffic_unit  df_site_revenue

# compute_site_margin   df_margin_per_site
 
#  compute_increase_of_yearly_site_margin   df_increase_arpu_by_year

# def compute_increase_cash_flow(df_capex, 
#                                df_sites, 
#                                df_increase_in_margin_due_to_the_upgrade, 
#                                selected_band_per_site, 
#                                output_route=conf['PATH']['MODELS_OUTPUT']):
    
    
    
    
    
 

        
        
 
    
    
    
    
compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='smartcapex-pipeline.json')

start_pipeline = pipeline_jobs.PipelineJob(
    display_name="smartcapex-pipeline",
    template_path="smartcapex-pipeline.json",
     enable_caching=False,
    parameter_values={
        "project": "osn-smartcapex-404-sbx",
        "dataset_src": "Dataset_SmartCapex",
        "table_site_raw": "Thies_referentielSite-OCI",   
        "table_oss_2g_raw": "osscounter2G-OCI", 
        "table_oss_3g_raw": "osscounter3G-OCI", 
        "table_oss_4g_raw": "osscounter4G-OCI", 
        "table_oss_rtx_raw": "trx2G-OCI",
        "table_typology_sector_raw": "typology_sector",
        "table_unit_prices": "unit_prices",        
        "table_charge_total_opex": "charge_total_opex",
        "table_aupu": "AUPU",
        "table_capex": "capex",
        "dataset_preprocessing": "preprocessing",
        "df_sites": "df_sites",
        "oss_counter_weekly": "oss_counter_weekly",
        "dataset_intermediate": "intermediate",
        "df_traffic_predicted":"df_predicted_traffic_kpis",
        "df_distance": "df_distance",
        "df_selected_band_per_site" :"df_selected_band_per_site_upgrade_selection" ,
        "df_affected_cells" :"df_affected_cells_upgrade_selection",
        "df_future_upgrades" :"df_future_upgrades_upgrade_selection",
        "df_cell_affected" :"df_cell_affected_traffic_improvement",
        "list_of_upgrades" :"list_of_upgrades_traffic_improvement",
        "sites_to_remove" :"sites_to_remove_traffic_improvement",
        "df_data_dl_traffic_features" :"df_data_dl_traffic_features_traffic_improvement",
        "df_data_ul_traffic_features" :"df_data_ul_traffic_features_traffic_improvement",
        "df_voice_traffic_features" :"df_voice_traffic_features_traffic_improvement",
        "df_traffic_by_region" :"df_traffic_by_region_traffic_improvement",
        "df_traffic_features_future_upgrades_data_dl" :"df_traffic_features_future_upgrades_data_dl_traffic_improvement",
        "df_traffic_features_future_upgrades_data_ul" :"df_traffic_features_future_upgrades_data_ul_traffic_improvement",
        "df_traffic_features_future_upgrades_voice" :"df_traffic_features_future_upgrades_voice_traffic_improvement",
        "df_traffic_features_future_upgrades_prediction_data_dl" :"df_traffic_features_future_upgrades_prediction_data_dl_traffic_improvement",
        "df_traffic_features_future_upgrades_prediction_data_ul" :"df_traffic_features_future_upgrades_prediction_data_ul_traffic_improvement",
        "df_traffic_features_future_upgrades_prediction_voice" :"df_traffic_features_future_upgrades_prediction_voice_traffic_improvement",
        "df_increase_traffic_after_upgrade_data_dl": "df_increase_traffic_after_upgrade_data_dl_traffic_improvement",
        "df_increase_traffic_after_upgrade_data_ul": "df_increase_traffic_after_upgrade_data_ul_traffic_improvement",
        "df_increase_traffic_after_upgrade_voice": "df_increase_traffic_after_upgrade_voice_traffic_improvement",
        "df_predicted_increase_in_traffic_by_the_upgrade": "df_predicted_increase_in_traffic_by_the_upgrade_traffic_improvement",
        "df_increase_of_arpu_by_the_upgrade": "df_increase_of_arpu_by_the_upgrade_arpu_quantification",
        "df_site_revenue": "df_site_revenue_compute_revenu_per_traffic_unit",
        "df_margin_per_site": "df_margin_per_site_compute_site_margin",
        "df_increase_arpu_by_year": "df_increase_arpu_by_year_compute_increase_of_yearly_site_margin_o",
        "df_final": "df_final_compute_increase_cash_flow", 
        "df_final_npv_of_the_upgrade": "df_final_npv_of_the_upgrade",
    }
)

print ("666666666666666666666666 Before run call 66666666666666666666666")
start_pipeline.run(service_account="sa-oro-pipeline-smart-capex@oro-smart-capex-001-dev.iam.gserviceaccount.com")
 