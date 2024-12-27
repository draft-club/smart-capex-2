from kfp import dsl
from utils.config import pipeline_config
from components.step_01_load_traffic_weekly_kpis import load_traffic_weekly_kpis
from components.step_02_load_affected_cells_data import load_affected_cells_data
from components.step_03_load_list_of_upgrades_data import load_list_of_upgrades_data
from components.step_04_load_sites_to_remove_data import load_sites_to_remove_data
from components.step_05_compute_upgrade_typology_features import compute_upgrade_typology_features
from components.step_06_compute_neighbors_of_upgrades import compute_neighbors_of_upgrades
from components.step_07_compute_traffic_per_site_and_tech import compute_traffic_per_site_and_tech
from components.step_08_compute_traffic_per_cluster_and_tech import compute_traffic_per_cluster_and_tech
from components.step_09_compute_traffic_model_features import compute_traffic_model_features
from components.step_10_get_capacity_kpis_features_model import get_capacity_kpis_features_model
from components.step_11_merge_all_improvement_features import merge_all_improvement_features
from components.step_12_filter_data_voice_traffic_features import filter_data_voice_traffic_features


@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description="Smart Capex: Get All Traffic Improvement Pipeline")

def pipeline(project_id: str,
             location: str,
             traffic_weekly_kpis_table_id: str,
             data_traffic_features_table_id:str,
             voice_traffic_features_table_id:str,
             max_number_of_neighbors: int,
             cell_affected_table_id: str,
             sites_to_remove_table_id: str,
             list_of_upgrades_table_id: str,
             bands_to_consider: list,
             weeks_to_wait_after_upgrade:int,
             weeks_to_wait_after_upgrade_max:int,
            ):
    """_summary_

    Args:
        project_id (str): _description_
        location (str): _description_
        traffic_weekly_kpis_table_id (str): _description_
        data_traffic_features_table_id (str): _description_
        voice_traffic_features_table_id (str): _description_
        max_number_of_neighbors (int): _description_
        cell_affected_table_id (str): _description_
        sites_to_remove_table_id (str): _description_
        list_of_upgrades_table_id (str): _description_
        bands_to_consider (list): _description_
        weeks_to_wait_after_upgrade (int): _description_
        weeks_to_wait_after_upgrade_max (int): _description_
    """

    load_traffic_weekly_kpis_data_op = load_traffic_weekly_kpis(project_id=project_id,
                                                                location=location,
                                                                table_id=traffic_weekly_kpis_table_id
                                                                ).set_display_name("load_traffic_weekly_kpis")


    load_affected_cells_data_op = load_affected_cells_data(project_id=project_id,
                                                           location=location,
                                                           table_id=cell_affected_table_id
                                                           ).set_display_name("load_affected_cells_data")


    load_list_of_upgrades_data_op = load_list_of_upgrades_data(project_id=project_id,
                                                               location=location,
                                                               table_id=list_of_upgrades_table_id
                                                               ).set_display_name("load_list_of_upgrades_data")


    load_sites_to_remove_data_op = load_sites_to_remove_data(project_id=project_id,
                                                             location=location,
                                                             table_id=sites_to_remove_table_id
                                                             ).set_display_name("load_sites_to_remove_data")


    compute_upgrade_typology_features_op = compute_upgrade_typology_features(
        bands_to_consider=bands_to_consider,
        max_number_of_neighbors=max_number_of_neighbors,
        traffic_weekly_kpis_data_input=load_traffic_weekly_kpis_data_op.outputs['query_results_data_output'],
        list_of_upgrades_data_input=load_list_of_upgrades_data_op.outputs['query_results_data_output']
        ).set_display_name("compute_upgrade_typology_features")


    compute_neighbors_of_upgrades_op = compute_neighbors_of_upgrades(
        max_number_of_neighbors=max_number_of_neighbors,
        remove_sites_with_more_than_one_upgrade_same_cluster=True,
        list_of_upgrades_data_input=load_list_of_upgrades_data_op.outputs['query_results_data_output'],
        sites_to_remove_data_input=load_sites_to_remove_data_op.outputs['query_results_data_output']
        ).set_display_name("compute_neighbors_of_upgrades")


    compute_traffic_per_site_and_tech_data_op =  compute_traffic_per_site_and_tech(
        kpi_to_compute_upgrade_effect="total_data_traffic_dl_gb",
        traffic_weekly_kpis_data_input=load_traffic_weekly_kpis_data_op.outputs['query_results_data_output'],
        neighbors_of_upgrades_data_input=compute_neighbors_of_upgrades_op.outputs['neighbors_of_upgrades_data_output']
        ).set_display_name("compute_traffic_per_site_and_tech_data_traffic_dl")


    compute_traffic_per_site_and_tech_voice_op =  compute_traffic_per_site_and_tech(
        kpi_to_compute_upgrade_effect="total_voice_traffic_kerlands",
        traffic_weekly_kpis_data_input=load_traffic_weekly_kpis_data_op.outputs['query_results_data_output'],
        neighbors_of_upgrades_data_input=compute_neighbors_of_upgrades_op.outputs['neighbors_of_upgrades_data_output']
        ).set_display_name("compute_traffic_per_site_and_tech_voice_traffic")


    compute_traffic_per_cluster_and_tech_data_op = compute_traffic_per_cluster_and_tech(
        kpi_to_compute_upgrade_effect="total_data_traffic_dl_gb",
        traffic_weekly_kpis_site_data_input=compute_traffic_per_site_and_tech_data_op.outputs['traffic_weekly_kpis_site_data_output'],
        traffic_weekly_kpis_site_tech_data_input=compute_traffic_per_site_and_tech_data_op.outputs['traffic_weekly_kpis_site_tech_data_output']
        ).set_display_name("compute_traffic_per_cluster_and_tech_data_traffic_dl")


    compute_traffic_per_cluster_and_tech_voice_op = compute_traffic_per_cluster_and_tech(
        kpi_to_compute_upgrade_effect="total_voice_traffic_kerlands",
        traffic_weekly_kpis_site_data_input=compute_traffic_per_site_and_tech_voice_op.outputs['traffic_weekly_kpis_site_data_output'],
        traffic_weekly_kpis_site_tech_data_input=compute_traffic_per_site_and_tech_voice_op.outputs['traffic_weekly_kpis_site_tech_data_output']
        ).set_display_name("compute_traffic_per_cluster_and_tech_voice_traffic")


    compute_traffic_model_features_data_op = compute_traffic_model_features(
        compute_target=True,
        weeks_to_wait_after_upgrade=weeks_to_wait_after_upgrade,
        weeks_to_wait_after_upgrade_max=weeks_to_wait_after_upgrade_max,
        kpi_to_compute_upgrade_effect="total_data_traffic_dl_gb",
        traffic_weekly_kpis_cluster_data_input=compute_traffic_per_cluster_and_tech_data_op.outputs['traffic_weekly_kpis_cluster_data_output'],
        traffic_weekly_kpis_cluster_tech_data_input=compute_traffic_per_cluster_and_tech_data_op.outputs['traffic_weekly_kpis_cluster_tech_data_output']
        ).set_display_name("compute_traffic_model_features_data_traffic_dl")


    compute_traffic_model_features_voice_op = compute_traffic_model_features(
        compute_target=True,
        weeks_to_wait_after_upgrade=weeks_to_wait_after_upgrade,
        weeks_to_wait_after_upgrade_max=weeks_to_wait_after_upgrade_max,
        kpi_to_compute_upgrade_effect="total_voice_traffic_kerlands",
        traffic_weekly_kpis_cluster_data_input=compute_traffic_per_cluster_and_tech_voice_op.outputs['traffic_weekly_kpis_cluster_data_output'],
        traffic_weekly_kpis_cluster_tech_data_input=compute_traffic_per_cluster_and_tech_voice_op.outputs['traffic_weekly_kpis_cluster_tech_data_output']
        ).set_display_name("compute_traffic_model_features_voice_traffic")


    get_capacity_kpis_features_model_op = get_capacity_kpis_features_model(
        operation_to_aggregate_cells= 'mean',
        cell_affected_data_input=load_affected_cells_data_op.outputs["query_results_data_output"])


    merge_all_improvement_data_features_op = merge_all_improvement_features(
        upgraded_to_not_consider=['2G'],
        kpi_to_compute_upgrade_effect="total_data_traffic_dl_gb",
        upgrades_features_typology_data_input=compute_upgrade_typology_features_op.outputs['upgrades_features_data_output'],
        traffic_model_features_data_input=compute_traffic_model_features_data_op.outputs['traffic_features_data_output'],
        capacity_kpis_features_data_input=get_capacity_kpis_features_model_op.outputs['capacity_kpis_features_data_output'],
        traffic_site_tech_data_input=compute_traffic_model_features_data_op.outputs['traffic_site_tech_data_output'],
        list_of_upgrades_data_input= load_list_of_upgrades_data_op.outputs['query_results_data_output']
        ).set_display_name("merge_all_improvement_features_data_traffic_dl")


    merge_all_improvement_voice_features_op = merge_all_improvement_features(
        upgraded_to_not_consider=['2G'],
        kpi_to_compute_upgrade_effect="total_voice_traffic_kerlands",
        upgrades_features_typology_data_input=compute_upgrade_typology_features_op.outputs['upgrades_features_data_output'],
        traffic_model_features_data_input=compute_traffic_model_features_voice_op.outputs['traffic_features_data_output'],
        capacity_kpis_features_data_input=get_capacity_kpis_features_model_op.outputs['capacity_kpis_features_data_output'],
        traffic_site_tech_data_input=compute_traffic_model_features_voice_op.outputs['traffic_site_tech_data_output'],
        list_of_upgrades_data_input= load_list_of_upgrades_data_op.outputs['query_results_data_output']
        ).set_display_name("merge_all_improvement_features_voice_traffic")


    filter_data_voice_traffic_features(
        project_id=project_id, location=location,
        data_traffic_features_table_id=data_traffic_features_table_id,
        voice_traffic_features_table_id=voice_traffic_features_table_id,
        merged_data_traffic_features_data_input=merge_all_improvement_data_features_op.outputs['merged_features_data_output'],
        merged_voice_traffic_features_data_input=merge_all_improvement_voice_features_op.outputs['merged_features_data_output'])
