from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_sites_data import load_sites_data
from components.step_02_load_processed_traffic_kpis_data import load_processed_traffic_kpis_data
from components.step_03_compute_distance_between_sites import compute_distance_between_sites
from components.step_04_get_affected_cells_with_interactions_between_upgrades import (
    get_affected_cells_with_interactions_between_upgrades
)
from components.step_05_get_cluster_of_affected_sites import get_cluster_of_affected_sites


@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description="Smart Capex: Preprocess Technical Pipeline")

def pipeline(project_id: str,
             location: str,
             processed_sites_table_id: str,
             processed_oss_counter_table_id: str,
             maximum_weeks_to_group_upgrade:int,
             max_number_of_neighbour:int,
             cell_affected_table_id:str,
             sites_to_remove_table_id:str,
             list_of_upgrades_table_id:str):

    load_data_processed_sites_op = load_sites_data(project_id=project_id,
                                                   location=location,
                                                   table_id=processed_sites_table_id)

    load_data_processed_traffic_kpis_op = load_processed_traffic_kpis_data(project_id=project_id,
                                                                           location=location,
                                                                           table_id=processed_oss_counter_table_id)

    compute_distance_between_sites_op = compute_distance_between_sites(
    processed_sites_data_input=load_data_processed_sites_op.outputs['query_results_data_output'])

    get_affected_cells_with_interactions_between_upgrades_op = get_affected_cells_with_interactions_between_upgrades(
    project_id=project_id,
    location=location,
    cell_affected_table_id=cell_affected_table_id,
    maximum_weeks_to_group_upgrade=maximum_weeks_to_group_upgrade,
    processed_oss_counter_data_input=load_data_processed_traffic_kpis_op.outputs['query_results_data_output'])

    get_cluster_of_affected_sites_op = get_cluster_of_affected_sites(
    project_id=project_id,
    location=location,
    sites_to_remove_table_id=sites_to_remove_table_id,
    list_of_upgrades_table_id=list_of_upgrades_table_id,
    max_number_of_neighbour=max_number_of_neighbour,
    cell_affected_data_input=get_affected_cells_with_interactions_between_upgrades_op.outputs['cell_affected_data_output'],
    sites_distances_data_input=compute_distance_between_sites_op.outputs['sites_distances_data_output'])

    # compute_rate_increase_matrix_all_kpis_op = compute_rate_increase_matrix_all_kpis(
    # weeks_to_wait_after_upgrade=weeks_to_wait_after_upgrade,
    # list_of_upgrades_data_input=get_cluster_of_affected_sites_op.outputs['list_of_upgrades_data_output'],
    # processed_oss_counter_data_input=load_data_processed_traffic_kpis_op.outputs['query_results_data_output'])
