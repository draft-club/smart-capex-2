from kfp import dsl
from components.step_01_load_oss_counters import load_oss_counters
from components.step_02_load_sites import load_sites
from components.step_03_load_capex_opex import load_capex_opex
from components.step_04_preprocess_sites import preprocess_sites
from components.step_05_preprocess_cells_bands import preprocess_cells_bands
from components.step_06_preprocess_oss_counter import preprocess_oss_counter
from components.step_07_merge_oss_counters import merge_oss_counters
from components.step_08_remove_recent_cells import remove_recent_cells
from components.step_09_remove_unmounted_cells import remove_unmounted_cells
from components.step_10_remove_cells_with_missing_weeks import remove_cells_with_missing_weeks
from components.step_11_remove_cells_with_high_variation_in_kpi import remove_cells_with_high_variation_in_kpi
from components.step_12_merge_oss_counter_with_sites import merge_oss_counter_with_sites
from components.step_13_save_cells_not_to_consider import save_cells_not_to_consider
from components.step_14_preprocess_capex_opex import preprocess_capex_opex

from utils.config import pipeline_config

@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description="Data Preparation Pipeline")

# create the pipeline and define parameters
def pipeline(project_id: str,
             location: str,
             deployment_history_table_id: str,
             oss_3g_table_id: str,
             oss_3g_history_table_id: str,
             oss_4g_table_id: str,
             oss_4g_history_table_id: str,
             processed_sites_table_id: str,
             processed_oss_counter_table_id: str,
             cells_not_to_consider_table_id: str,
             recent_weeks_threshold: int,
             unmounted_weeks_threshold: int,
             missing_weeks_threshold: int,
             variation_coefficient_threshold: float,
             capex_table_id: str,
             opex_table_id: str,
             processed_capex_table_id: str,
             processed_opex_table_id: str,):

    """Data Preparation Vertex pipeline flow.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        deployment_history_table_id (str): It holds the deployment history table ID.
        oss_3g_table_id (str): It holds the OSS 3G table ID.
        oss_3g_history_table_id (str): It holds the OSS 3G history table ID.
        oss_4g_table_id (str): It holds the OSS 4G table ID.
        oss_4g_history_table_id (str): It holds the OSS 4G history table ID.
        processed_sites_table_id (str): It holds the Processed sites table ID.
        processed_oss_counter_table_id (str): It holds the processed OSS counter table ID.
        cells_not_to_consider_table_id (str): It holds the cells not to consider table ID.
        recent_weeks_threshold (int): It holds the threshold for the number of recent weeks.
        unmounted_weeks_threshold (int): It holds the threshold for the number of weeks with unmounted cells.
        missing_weeks_threshold (int): It holds the threshold for the number of missing weeks.
        variation_coefficient_threshold (float): It holds the threshold for variation coefficient.
    """


    load_sites_data_op = load_sites(project_id=project_id,
                                    location=location,
                                    deployment_history_table_id=deployment_history_table_id)

    load_oss_counters_3g_op = load_oss_counters(project_id=project_id,
                                                 location=location,
                                                 oss_counters_table_id=oss_3g_table_id,
                                                 oss_counters_history_table_id=oss_3g_history_table_id,
                                                 cell_technology='3G').set_display_name("load_oss_counter_3G")

    load_oss_counters_4g_op = load_oss_counters(project_id=project_id,
                                                 location=location,
                                                 oss_counters_table_id=oss_4g_table_id,
                                                 oss_counters_history_table_id=oss_4g_history_table_id,
                                                 cell_technology='4G').set_display_name("load_oss_counter_4G")
     
    load_capex_opex_op = load_capex_opex(project_id=project_id,
                                         location=location,
                                         capex_table_id=capex_table_id,
                                         opex_table_id=opex_table_id)

    preprocess_sites_data_op = preprocess_sites(raw_sites_data_input=load_sites_data_op.outputs["raw_sites_data_output"])

    preprocess_sites_cells_bands_data_op = preprocess_cells_bands(
        project_id=project_id,
        location=location,
        preprocessed_sites_data_input=preprocess_sites_data_op.outputs["processed_sites_data_output"],
        processed_sites_table_id=processed_sites_table_id)

    preprocess_oss_counter_3g_data_op = preprocess_oss_counter(
        cell_technology='3G',
        raw_oss_counter_data_input=load_oss_counters_3g_op.outputs["raw_oss_data_output"]
        ).set_display_name("preprocess_oss_counter_3G")

    preprocess_oss_counter_4g_data_op = preprocess_oss_counter(
        cell_technology='4G',
        raw_oss_counter_data_input=load_oss_counters_4g_op.outputs["raw_oss_data_output"]
        ).set_display_name("preprocess_oss_counter_4G")

    merge_oss_counters_data_op = merge_oss_counters(
        processed_3g_data_input=preprocess_oss_counter_3g_data_op.outputs["processed_oss_counter_data_output"],
        processed_4g_data_input=preprocess_oss_counter_4g_data_op.outputs["processed_oss_counter_data_output"])


    remove_recent_cells_data_op = remove_recent_cells(
        recent_weeks_threshold=recent_weeks_threshold,
        merged_oss_counters_data_input=merge_oss_counters_data_op.outputs['oss_counters_data_output'])

    remove_unmounted_cells_data_op = remove_unmounted_cells(
        unmounted_weeks_threshold=unmounted_weeks_threshold,
        oss_counter_data_without_recent_cells_data_input=remove_recent_cells_data_op.outputs['oss_counter_data_without_recent_cells_data_output'])

    remove_cells_with_missing_weeks_data_op = remove_cells_with_missing_weeks(
        missing_weeks_threshold=missing_weeks_threshold,
        oss_counter_data_without_unmounted_cells_data_input=remove_unmounted_cells_data_op.outputs['oss_counter_data_without_unmounted_cells_data_output'])

    remove_cells_with_high_variation_in_kpi_data_op = remove_cells_with_high_variation_in_kpi(
        variation_coefficient_threshold=variation_coefficient_threshold,
        oss_counter_data_without_missing_weeks_data_input=remove_cells_with_missing_weeks_data_op.outputs['oss_counter_data_without_missing_weeks_data_output'])

    merge_oss_counter_with_sites(
        project_id=project_id,
        location=location,
        processed_oss_counter_table_id=processed_oss_counter_table_id,
        oss_counter_data_without_high_variation_data_input=remove_cells_with_high_variation_in_kpi_data_op.outputs['oss_counter_data_without_high_variation_in_kpi_data_output'],
        processed_sites_data_input=preprocess_sites_cells_bands_data_op.outputs["processed_sites_cells_bands_data_output"])

    # Decided to calculate it on the fly for the next pipelines due to its large size (over 49 Million records)
    save_cells_not_to_consider(
        project_id=project_id,
        location=location,
        cells_not_to_consider_table_id=cells_not_to_consider_table_id,
        cells_not_to_consider_recent_cells=remove_recent_cells_data_op.outputs['cells_not_to_consider_data_output'],
        cells_not_to_consider_unmounted_cells=remove_unmounted_cells_data_op.outputs['cells_not_to_consider_data_output'],
        cells_not_to_consider_with_missing_weeks=remove_cells_with_missing_weeks_data_op.outputs['cells_not_to_consider_data_output'],
        cells_not_to_consider_with_high_variation_in_kpi=remove_cells_with_high_variation_in_kpi_data_op.outputs['cells_not_to_consider_data_output'])
    
    preprocess_capex_opex_op = preprocess_capex_opex(project_id=project_id,
                                                     location=location,
                                                     processed_capex_table_id=processed_capex_table_id,
                                                     processed_opex_table_id=processed_opex_table_id,
                                                     capex_data_input=load_capex_opex_op.outputs["capex_data_output"],
                                                     opex_data_input=load_capex_opex_op.outputs["opex_data_output"])
