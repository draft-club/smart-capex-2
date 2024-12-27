from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_revenues_per_unit_traffic_data import load_revenues_per_unit_traffic_data
from components.step_02_load_increase_arpu_due_to_the_upgrade_data import load_increase_arpu_due_to_the_upgrade_data
from components.step_03_load_opex_data import load_opex_data
from components.step_04_load_capex_data import load_capex_data
from components.step_05_compute_site_margin import compute_site_margin
from components.step_06_compute_increase_of_yearly_site_margin import compute_increase_of_yearly_site_margin
from components.step_07_compute_increase_cash_flow import compute_increase_cash_flow
from components.step_08_compute_npvs import compute_npvs

@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description=pipeline_config["description"])
def pipeline(project_id: str,
             location: str,
             revenues_per_unit_traffic_table_id: str,
             increase_arpu_due_to_the_upgrade_table_id: str,
             processed_opex_table_id: str,
             processed_capex_table_id: str,
             margin_per_site_table_id: str,
             increase_arpu_by_year_table_id: str,
             cash_flow_table_id: str,
             npv_of_the_upgrade_table_id: str,
             time_to_compute_npv: int,
             wacc: int
            ):
    """Define the pipeline to calculate the Net Present Value (NPV) of site upgrades

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        revenues_per_unit_traffic_table_id (str): It holds BigQuery table ID for revenues per unit traffic
        increase_arpu_due_to_the_upgrade_table_id (str): It holds BigQuery table ID for increase arpu
        processed_opex_table_id (str): It holds BigQuery table ID for opex
        processed_capex_table_id (str): It holds BigQuery table ID for capex
        margin_per_site_table_id (str): It holds BigQuery table ID for margin per site
        increase_arpu_by_year_table_id (str): It holds BigQuery table ID for increase arpu by year
        cash_flow_table_id (str): It holds BigQuery table ID for cash flow
        npv_of_the_upgrade_table_id (str): It holds BigQuery table ID for npv of the upgrade
        time_to_compute_npv (int): It holds the number of years to compute the Net Present Value (NPV).
        wacc (int): It holds the Weighted Average Cost of Capital (WACC) used for discounting cash flows
    """

    load_revenues_per_unit_traffic_data_op = load_revenues_per_unit_traffic_data(project_id=project_id,
                                                                                 location=location,
                                                                                 table_id=revenues_per_unit_traffic_table_id)

    load_increase_arpu_due_to_the_upgrade_data_op = load_increase_arpu_due_to_the_upgrade_data(
        project_id=project_id,
        location=location,
        table_id=increase_arpu_due_to_the_upgrade_table_id)

    load_opex_data_op = load_opex_data(project_id=project_id,
                                       location=location,
                                       table_id=processed_opex_table_id)

    load_capex_data_op = load_capex_data(project_id=project_id,
                                         location=location,
                                         table_id=processed_capex_table_id)


    compute_site_margin_op = compute_site_margin(
        project_id=project_id,
        location=location,
        margin_per_site_table_id=margin_per_site_table_id,
        revenues_per_unit_traffic_data_input=load_revenues_per_unit_traffic_data_op.outputs["query_results_data_output"])

    compute_increase_of_yearly_site_margin_op = compute_increase_of_yearly_site_margin(
        project_id=project_id,
        location=location,
        increase_arpu_by_year_table_id=increase_arpu_by_year_table_id,
        revenues_per_unit_traffic_data_input=load_revenues_per_unit_traffic_data_op.outputs["query_results_data_output"],
        increase_arpu_due_to_the_upgrade_data_input=\
            load_increase_arpu_due_to_the_upgrade_data_op.outputs["query_results_data_output"],
        margin_per_site_data_input=compute_site_margin_op.outputs["margin_per_site_data_output"])

    compute_increase_cash_flow_op = compute_increase_cash_flow(
        project_id=project_id,
        location=location,
        cash_flow_table_id=cash_flow_table_id,
        time_to_compute_npv=time_to_compute_npv,
        opex_data_input=load_opex_data_op.outputs["query_results_data_output"],
        capex_data_input=load_capex_data_op.outputs["query_results_data_output"],
        increase_in_margin_due_to_the_upgrade_data_input=\
            compute_increase_of_yearly_site_margin_op.outputs["increase_arpu_by_year_data_output"])

    compute_npvs_op = compute_npvs(project_id=project_id,
                                   location=location,
                                   npv_of_the_upgrade_table_id=npv_of_the_upgrade_table_id,
                                   wacc=wacc,
                                   df_cash_flow_data_input=compute_increase_cash_flow_op.outputs["cash_flow_data_output"])
