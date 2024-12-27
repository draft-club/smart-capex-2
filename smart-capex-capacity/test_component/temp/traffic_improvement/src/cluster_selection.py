
import logging
logger = logging.getLogger("my logger")
import pandas as pd
#from src.d03_capacity.technical_modules.activation_model import *
from src.activation_model import *
import numpy as np
from scipy import spatial
import haversine as hs
#from src.d00_conf.conf import conf, conf_loader
from src.conf import conf, conf_loader
conf_loader("OSN")
# from src.d01_utils.utils import *

def compute_distance_between_sites(df_sites):
    
    df_sites['site_longitude'] = df_sites['site_longitude'].str.replace(',', '.').astype(float)
    df_sites['site_latitude'] = df_sites['site_latitude'].str.replace(',', '.').astype(float)

    df_sites = df_sites[['site_longitude','site_latitude', 'site_id']].drop_duplicates()
    df_nan = df_sites.site_latitude.isna()
    df_sites = df_sites[~df_nan].drop_duplicates()
    df_sites = df_sites.groupby('site_id').first().reset_index(drop = False)

    all_points = df_sites[['site_latitude', 'site_longitude']].values


    dm1 = spatial.distance.cdist(all_points,
                                 all_points,
                                 hs.haversine)

    df_distance = pd.DataFrame(dm1, index =  df_sites['site_id'].values, columns = df_sites['site_id'].values)

    return df_distance


def compute_list_neighbours_sites(site_id,
                                  bands_upgraded,
                                  df_distance,
                                  max_number = 2):
    try:
        distance_vector = df_distance[site_id]
    except:
        return None

    if 'L8' in bands_upgraded or 'U09' in bands_upgraded:
        distance = 3.5
    else:
        distance = 2

    distance_vector  = distance_vector[(distance_vector > 0) & (distance_vector < distance)]
    return distance_vector.sort_values()[0:max_number].index

def compute_neighbour_site(site_id,
                           bands_upgraded,
                           df_distance,
                           position):
    try:
        distance_vector = df_distance[site_id]
    except:
        return ""

    if 'L8' in bands_upgraded or 'U9' in bands_upgraded:
        distance = conf['TRAFFIC_IMPROVEMENT']['DISTANCES']['TOTAL']['L8']
    else:
        distance = conf['TRAFFIC_IMPROVEMENT']['DISTANCES']['TOTAL']['L26']
    distance_vector  = distance_vector[(distance_vector > 0) & (distance_vector < distance)].sort_values()

    if position > len(distance_vector):
        return ""
    else:
        return distance_vector.index[position-1]

def get_cluster_of_affected_sites(df_cell_affected,
                                  df_distance,
                                  max_neighbors = conf['TRAFFIC_IMPROVEMENT']['MAX_NUMBER_OF_NEIGHBORS']):

    """
    Function that gets all the data needed to train the capacity KPIs model

    """
    list_of_upgrades  = df_cell_affected[['site_id',
                                          'week_of_the_upgrade',
                                          'bands_upgraded',
                                          'tech_upgraded']].drop_duplicates()

    list_of_upgrades['cluster_key'] = list_of_upgrades['site_id'].apply(str) + "-" + list_of_upgrades['week_of_the_upgrade'].apply(str)

    for i in range(0, max_neighbors):
        list_of_upgrades['neighbor_' + str(i+1)] = list_of_upgrades[['site_id','bands_upgraded']].\
                                                                                                apply(lambda x: compute_neighbour_site(x.iloc[0],
                                                                                                           x.iloc[1],
                                                                                                           df_distance,
                                                                                                           i+1), axis =1)
        list_of_upgrades.fillna("", inplace=True)
        list_of_upgrades['cluster_key'] =list_of_upgrades['cluster_key'] + "-" + list_of_upgrades['neighbor_' + str(i+1)]


    neighbors = {}
    sites_to_remove = []
    for index, row in list_of_upgrades.iterrows():
        neighbors_sites  = compute_list_neighbours_sites(row[0],
                                                    row[2],
                                                    df_distance)
        if (neighbors_sites is None):
            row['neighbors'] = [""]
            neighbors[row[0]] = []
            continue

        neighbors[row[0]] = neighbors_sites.values
        ## Compute if there is an upgrade on sites on the same cluster in a period less than 2 months
        for site in neighbors_sites:
            if site in list_of_upgrades['site_id'].values:
                upgrade= list_of_upgrades[list_of_upgrades['site_id']==site]
                upgrade_week = upgrade['week_of_the_upgrade'].values[0]
                if abs(get_lag_between_two_week_periods(str(row[1]), str(upgrade_week))) <=8:
                    if row['site_id'] not in sites_to_remove:
                        sites_to_remove.append(row['site_id'])

    return list_of_upgrades, sites_to_remove



def get_cluster_of_future_upgrades(selected_band_per_site,
                                  df_distance,
                                  max_neighbors = conf['TRAFFIC_IMPROVEMENT']['MAX_NUMBER_OF_NEIGHBORS']):

    """
    Function that gets all the data needed to train the capacity KPIs model

    """
    print('columns of cluster future ',selected_band_per_site.columns)
    selected_band_per_site['cluster_key'] = selected_band_per_site['site_id']+'-'+selected_band_per_site['week_of_the_upgrade'].apply(str)
    #selected_band_per_site['cluster_key'] = selected_band_per_site['site_id']
    for i in range(0, max_neighbors):
        selected_band_per_site['neighbor_' + str(i+1)] = selected_band_per_site[['site_id','bands_upgraded']].apply(lambda x: compute_neighbour_site(
                                                                                                                        x.iloc[0],
                                                                                                                        x.iloc[1],
                                                                                                                        df_distance,
                                                                                                                        i+1), axis =1)

        selected_band_per_site['cluster_key'] = selected_band_per_site['cluster_key'] + "-" + selected_band_per_site['neighbor_' + str(i+1)]

    # selected_band_per_site.to_csv("/data/OSN/02_intermediate/file_for_test/selected_band_per_site_cluster.csv", sep="|",
    #                               index=False)

    return selected_band_per_site
