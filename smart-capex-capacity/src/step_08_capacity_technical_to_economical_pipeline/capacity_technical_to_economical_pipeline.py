from kfp import dsl

from utils.config import pipeline_config
from components.step_01_load_data import load_data
from components.step_02_compute_traffic_yearly_kpis import compute_traffic_yearly_kpis
from components.step_03_compute_revenues_per_site import compute_revenues_per_site
from components.step_04_compute_month_period import compute_month_period
from components.step_05_compute_unit_price_by_year import compute_unit_price_by_year
from components.step_06_compute_increase_of_arpu_by_the_upgrade import compute_increase_of_arpu_by_the_upgrade


@dsl.pipeline(
    # Default pipeline root
    name=pipeline_config["display_name"],
    pipeline_root=pipeline_config["pipeline_root"],
    description=pipeline_config["description"])

# create the pipeline and define parameters
def pipeline(project_id: str,
             location: str,
             traffic_weekly_kpis_table_id: str,
             unit_price_table_id: str,
             revenues_per_site_table_id: str,
             increase_of_arpu_by_the_upgrade_table_id: str,
             predicted_increase_in_traffic_by_the_upgrade_table_id: str,
             last_year_of_oss: int,
             opex_comissions_percentage: float,
             time_to_compute_npv: int):
    """
    Define the pipeline to compute various revenues per site and ARPU increases.

    Args:
        project_id (str): It holds GCP project ID.
        location (str): It holds the location of the BigQuery tables.
        traffic_weekly_kpis_table_id (str): It holds BigQuery table ID for weekly traffic KPIs.
        unit_price_table_id (str): It holds BigQuery table ID for unit prices.
        revenues_per_site_table_id (str): It holds BigQuery table ID for revenues per site.
        increase_of_arpu_by_the_upgrade_table_id (str): It holds BigQuery table ID for ARPU increase by the upgrade.
        predicted_increase_in_traffic_by_the_upgrade_table_id (str): It holds BigQuery table ID for predicted 
                                                                        increase in traffic by the upgrade.
        last_year_of_oss (int): It holds the last year of OSS.
        opex_comissions_percentage (float): It holds the percentage of OPEX commissions to be applied.
        time_to_compute_npv (int): It holds the number of years to compute NPV.
    """

    load_traffic_weekly_kpis_op = load_data(project_id=project_id,
                                            location=location,
                                            table_id=traffic_weekly_kpis_table_id
                                            ).set_display_name("load-traffic-weekly-kpis")

    load_unit_prices_op = load_data(project_id=project_id,
                                    location=location,
                                    table_id=unit_price_table_id).set_display_name("load-unit-prices")

    load_predicted_increase_in_traffic_by_the_upgrade_op = load_data(
                                                    project_id=project_id,
                                                    location=location,
                                                    table_id=predicted_increase_in_traffic_by_the_upgrade_table_id
                                                    ).set_display_name("load-predicted-increase-in-traffic-by-the-upgrade")

    compute_traffic_yearly_kpis_op = compute_traffic_yearly_kpis(
                            traffic_weekly_kpis_data_input=load_traffic_weekly_kpis_op.outputs["query_results_data_output"],
                            unit_prices_data_input=load_unit_prices_op.outputs["query_results_data_output"])

    compute_revenues_per_site(
                opex_comissions_percentage=opex_comissions_percentage,
                revenues_per_site_table_id=revenues_per_site_table_id,
                project_id = project_id,
                location=location,
                traffic_yearly_kpis_data_input=compute_traffic_yearly_kpis_op.outputs["traffic_yearly_kpis_data_output"])

    compute_month_period_op = compute_month_period(
        predicted_increase_in_traffic_by_the_upgrade_data_input=\
                                load_predicted_increase_in_traffic_by_the_upgrade_op.outputs["query_results_data_output"]
                                )

    compute_unit_price_by_year_op = compute_unit_price_by_year(
            time_to_compute_npv=time_to_compute_npv,
            last_year_of_oss=last_year_of_oss,
            unit_prices_data_input=load_unit_prices_op.outputs["query_results_data_output"],
            predicted_increase_in_traffic_by_the_upgrade_data_input=compute_month_period_op.outputs[
                                                                "predicted_increase_in_traffic_by_the_upgrade_data_output"])

    compute_increase_of_arpu_by_the_upgrade(
        increase_of_arpu_by_the_upgrade_table_id=increase_of_arpu_by_the_upgrade_table_id,
        project_id=project_id,
        location=location,
        predicted_increase_in_traffic_by_the_upgrade_data_input=compute_unit_price_by_year_op.outputs[
                                                                "predicted_increase_in_traffic_by_the_upgrade_data_output"])
