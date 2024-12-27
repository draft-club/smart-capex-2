from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_sites_data import load_sites_data
from components.step_02_load_predicted_traffic_kpis_data import load_predicted_traffic_kpis_data
from components.step_03_load_b32_data import load_b32_data
from components.step_04_calculate_compatible_terminals_ratio import calculate_compatible_terminals_ratio
from components.step_05_compute_distance_between_sites import compute_distance_between_sites
from components.step_06_add_unique_site_features import add_unique_site_features
from components.step_07_detect_cell_congestion import detect_cell_congestion
from components.step_08_handle_congested_cells import handle_congested_cells
from components.step_09_handle_no_congested_cells import handle_no_congested_cells
from components.step_10_post_process_upgraded_bands import post_process_upgraded_bands
from components.step_11_merge_predicted_kpis_with_bands import merge_predicted_kpis_with_bands
from components.step_12_get_cluster_of_future_upgrades import get_cluster_of_future_upgrades
from components.step_13_merge_congestion_with_sites import merge_congestion_with_sites

@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description="Smart Capex: Process Bands to Upgrade Pipeline")

# create the pipeline and define parameters
def pipeline(project_id: str,
             location: str,
             processed_sites_table_id: str,
             predicted_traffic_kpis_table_id: str,
             b32_table_id: str,
             week_of_the_upgrade: str,
             max_number_of_neighbors: int,
             selected_band_per_site_table_id: str,
             affected_cells_table_id: str,
             cluster_of_the_upgrade_table_id: str,
             congestion_status_for_db_table_id: str
            ):

    load_data_processed_sites_op = load_sites_data(project_id=project_id,
                                                   location=location,
                                                   table_id=processed_sites_table_id)

    load_data_predicted_traffic_kpis_op = load_predicted_traffic_kpis_data(project_id=project_id,
                                                                           location=location,
                                                                           table_id=predicted_traffic_kpis_table_id)

    load_data_b32_op = load_b32_data(project_id=project_id,
                                     location=location,
                                     table_id=b32_table_id)

    calculate_compatible_terminals_op = calculate_compatible_terminals_ratio(
        b32_data_input=load_data_b32_op.outputs["query_results_data_output"])

    add_unique_site_features_op = add_unique_site_features(
        processed_sites_data_input=load_data_processed_sites_op.outputs["query_results_data_output"])

    detect_cell_congestion_op = detect_cell_congestion(
        predicted_traffic_kpis_data_input=load_data_predicted_traffic_kpis_op.outputs["query_results_data_output"])

    handle_congested_cells_op = handle_congested_cells(
        week_of_the_upgrade=week_of_the_upgrade,
        b32_aggregated_data_input=calculate_compatible_terminals_op.outputs["b32_aggregated_data_output"],
        unique_site_features_data_input=add_unique_site_features_op.outputs["unique_site_features_data_output"],
        detected_cell_congestion_data_input=detect_cell_congestion_op.outputs["detected_cell_congestion_data_output"])

    handle_no_congested_cells_op = handle_no_congested_cells(
        unique_site_features_data_input=add_unique_site_features_op.outputs["unique_site_features_data_output"],
        detected_cell_congestion_data_input=detect_cell_congestion_op.outputs["detected_cell_congestion_data_output"],
        unique_congested_cells_data_input=handle_congested_cells_op.outputs["unique_congested_cells_data_output"])

    post_process_upgraded_bands_op = post_process_upgraded_bands(
        project_id=project_id,
        location=location,
        selected_band_per_site_table_id=selected_band_per_site_table_id,
        week_of_the_upgrade=week_of_the_upgrade,
        congestion_data_input=handle_congested_cells_op.outputs["congestion_data_output"],
        no_congestion_data_input=handle_no_congested_cells_op.outputs["no_congestion_data_output"])

    merge_predicted_kpis_with_bands_op = merge_predicted_kpis_with_bands(
        project_id=project_id, location=location,
        affected_cells_table_id=affected_cells_table_id,
        predicted_traffic_kpis_data_input=load_data_predicted_traffic_kpis_op.outputs["query_results_data_output"],
        selected_band_per_site_data_input=post_process_upgraded_bands_op.outputs["selected_band_per_site_data_output"])

    calculate_sites_distances_op = compute_distance_between_sites(
        processed_sites_data_input=load_data_processed_sites_op.outputs["query_results_data_output"])

    get_cluster_of_future_upgrades_op = get_cluster_of_future_upgrades(
        project_id=project_id,
        location=location,
        cluster_of_the_upgrade_table_id=cluster_of_the_upgrade_table_id,
        week_of_the_upgrade=week_of_the_upgrade,
        max_number_of_neighbors=max_number_of_neighbors,
        sites_distances_data_input=calculate_sites_distances_op.outputs["sites_distances_data_output"],
        selected_band_per_site_data_input=post_process_upgraded_bands_op.outputs["selected_band_per_site_data_output"])

    merge_congestion_with_sites_op = merge_congestion_with_sites(
        project_id=project_id,
        location=location,
        congestion_status_for_db_table_id=congestion_status_for_db_table_id,
        congestion_status_data_input=post_process_upgraded_bands_op.outputs["congestion_status_data_output"],
        sites_data_input=load_data_processed_sites_op.outputs["query_results_data_output"])
