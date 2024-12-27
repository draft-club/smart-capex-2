
from google.cloud import bigquery
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
from src.conf import conf, conf_loader
from src.activation_model import get_affected_cells_with_interactions_between_upgrades
from src.cluster_selection import get_cluster_of_affected_sites
from src.traffic_improvement import get_all_traffic_improvement_features, compute_traffic_after, train_traffic_improvement_model
from src.traffic_improvement_trend import compute_traffic_by_region, train_trend_model_with_linear_regression


conf_loader('OSN')

# import parameters
parser = argparse.ArgumentParser()
parser.add_argument('--PROJECT_ID', dest = 'PROJECT_ID', type = str)
parser.add_argument('--DATANAME', dest = 'DATANAME', type = str)
parser.add_argument('--NOTEBOOK', dest = 'NOTEBOOK', type = str)

args = parser.parse_args()
PROJECT_ID = args.PROJECT_ID
DATANAME = args.DATANAME
NOTEBOOK = args.NOTEBOOK

#my_arg = args.my_arg
print(PROJECT_ID, DATANAME, NOTEBOOK)

# client for BQ
bq = bigquery.Client(project = PROJECT_ID)

df_traffic_weekly_kpis = bq.query(query = f"SELECT * FROM `osn-smartcapex-404-sbx.preprocessing.oss_counter_weekly`" ).to_dataframe()
df_distance = bq.query(query = f"SELECT * FROM `osn-smartcapex-404-sbx.intermediate.df_distance`" ).to_dataframe()
df_sites = bq.query(query = f"SELECT * FROM `osn-smartcapex-404-sbx.preprocessing.df_sites`" ).to_dataframe()



# Script
df_traffic_weekly_kpis = df_traffic_weekly_kpis.replace({'cell_band': {'F2_U2100': 'U2100', 'F3_U2100': 'U2100', 'F1_U900': 'U900',
                                                                           'F1_U2100':'U2100','F4_U2100':'U2100','F2_U900':'U900'}})
    
df_cell_affected = get_affected_cells_with_interactions_between_upgrades(df_traffic_weekly_kpis)

list_of_upgrades, sites_to_remove = get_cluster_of_affected_sites(df_cell_affected,
                                                                    df_distance,
                                                                    max_neighbors=
                                                                    conf['TRAFFIC_IMPROVEMENT'][
                                                                    'MAX_NUMBER_OF_NEIGHBORS'])

df_data_traffic_features = get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                                                       df_cell_affected,
                                                                       list_of_upgrades,
                                                                       sites_to_remove,
                                                                       upgraded_to_not_consider=[])
df_data_traffic_features = compute_traffic_after(df_data_traffic_features, df_traffic_weekly_kpis, 'total_data_traffic_dl_gb')
df_voice_traffic_features = get_all_traffic_improvement_features(df_traffic_weekly_kpis,
                                                                       df_cell_affected,
                                                                       list_of_upgrades,
                                                                       sites_to_remove,
                                                                       type_of_traffic='voice',
                                                                       kpi_to_compute_upgrade_effect=[
                                                                           "total_voice_traffic_kerlands"],
                                                                       upgraded_to_not_consider=[])
df_voice_traffic_features = compute_traffic_after(df_voice_traffic_features, df_traffic_weekly_kpis,'total_voice_traffic_kerlands')
                                                                    
model_rf_data = train_traffic_improvement_model(df_data_traffic_features,
                                type_of_traffic='data',
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
                                                 'total_voice_traffic_kerlands'])

train_trend_model_with_linear_regression(df_traffic_by_region,
                                         variable_to_group_by=['site_region'],
                                         kpi_to_compute_trend=['total_data_traffic_dl_gb'])
train_trend_model_with_linear_regression(df_traffic_by_region,
                                         variable_to_group_by=['site_region'],
                                         kpi_to_compute_trend=['total_voice_traffic_kerlands'])



# output data - to BQ
df_cell_affected.to_gbq(f"{PROJECT_ID}.{DATANAME}.df_cell_affected", f'{PROJECT_ID}', if_exists = 'replace')
list_of_upgrades.to_gbq(f"{PROJECT_ID}.{DATANAME}.list_of_upgrades", f'{PROJECT_ID}', if_exists = 'replace')
print(sites_to_remove)
df_data_traffic_features.to_gbq(f"{PROJECT_ID}.{DATANAME}.df_data_traffic_features", f'{PROJECT_ID}', if_exists = 'replace')
df_voice_traffic_features.to_gbq(f"{PROJECT_ID}.{DATANAME}.df_voice_traffic_features", f'{PROJECT_ID}', if_exists = 'replace')
df_traffic_by_region.to_gbq(f"{PROJECT_ID}.{DATANAME}.df_traffic_by_region", f'{PROJECT_ID}', if_exists = 'replace')

