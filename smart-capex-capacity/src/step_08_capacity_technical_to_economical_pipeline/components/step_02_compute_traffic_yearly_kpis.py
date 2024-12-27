from kfp.dsl import Dataset, Input, Output, component

from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_traffic_yearly_kpis(traffic_weekly_kpis_data_input: Input[Dataset],
                                unit_prices_data_input: Input[Dataset],
                                traffic_yearly_kpis_data_output: Output[Dataset]):
    """
    Compute yearly KPIs from weekly traffic data and unit prices and save the output for Vertex pipeline.

    Args:
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the input dataset containing weekly traffic KPIs DataFrame.
        unit_prices_data_input (Input[Dataset]): It holds the input dataset containing unit prices 
                                                    for data and voice traffic.
        traffic_yearly_kpis_data_output (Output[Dataset]): It holds the output dataset to store 
                                                            the computed yearly KPIs DataFrame.
    """
    import pandas as pd

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    df_unit_prices = pd.read_parquet(unit_prices_data_input.path)

    kpis_to_compute = ['total_data_traffic_dl_gb', 'total_voice_traffic_kerlands']

    df_traffic_weekly_kpis["year"] = df_traffic_weekly_kpis["year"].astype(str)

    df_traffic_sum_all_cells = df_traffic_weekly_kpis.groupby(
                                                ['site_id', 'week_period', 'year'])[kpis_to_compute].sum().reset_index()

    df_traffic_avg_weekly_per_site = df_traffic_sum_all_cells.copy()
    df_traffic_avg_weekly_per_site[kpis_to_compute] = df_traffic_avg_weekly_per_site.groupby("site_id")[
                                                                                        kpis_to_compute].transform("mean")

    df_traffic_yearly_kpis = df_traffic_avg_weekly_per_site.copy()
    df_traffic_yearly_kpis[kpis_to_compute] = df_traffic_avg_weekly_per_site[kpis_to_compute] * 52

    df_traffic_yearly_kpis = pd.merge(left=df_traffic_yearly_kpis, right=df_unit_prices, on=["year"])

    df_traffic_yearly_kpis["revenue_voice_mobile"] = df_traffic_yearly_kpis["total_voice_traffic_kerlands"] * \
                                                        df_traffic_yearly_kpis["ppm_voice"] * 1000 * 60

    df_traffic_yearly_kpis["revenue_data_mobile"] = df_traffic_yearly_kpis["total_data_traffic_dl_gb"] * \
                                                        df_traffic_yearly_kpis["ppm_data"]

    df_traffic_yearly_kpis.to_parquet(traffic_yearly_kpis_data_output.path)
