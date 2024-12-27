from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_traffic_by_region(kpis_to_compute_trend: list,
                              traffic_weekly_kpis_data_input: Input[Dataset],
                              region_traffic_data_output: Output[Dataset]):
    """Compute traffic by region and save the resulted dataframe to a parquet file for vertex pipeline.

    Args:
        kpis_to_compute_trend (list): It holds the list of KPIs to compute the trend.
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the input dataset containing traffic weekly KPIs.
        region_traffic_data_output (Output[Dataset]): It holds the output dataset to store the region traffic data.
    """

    import pandas as pd

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)

    df_traffic_weekly_kpis["week_date"] = pd.to_datetime(df_traffic_weekly_kpis["week_date"])
    df_traffic_weekly_kpis.rename(columns={"region": "site_region"}, inplace=True)
    df = (df_traffic_weekly_kpis.groupby(["site_region", "week_date"])[kpis_to_compute_trend].sum().reset_index())
    df_list_date = list(df.week_date.unique())[:-1]
    df = df[df.week_date.isin(df_list_date)]

    df.to_parquet(region_traffic_data_output.path, index=False)
