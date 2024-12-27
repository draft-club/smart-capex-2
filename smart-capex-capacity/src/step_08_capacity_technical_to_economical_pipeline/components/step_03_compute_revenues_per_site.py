from kfp.dsl import Dataset, Input, component

from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_revenues_per_site(opex_comissions_percentage: float,
                              revenues_per_site_table_id: str,
                              project_id: str,
                              location: str,
                              traffic_yearly_kpis_data_input: Input[Dataset]):
    """
    Compute revenues per site and upload the results to a BigQuery table for other vertex pipelines.

    Args:
        opex_comissions_percentage (float): It holds percentage of OPEX commissions to be applied.
        revenues_per_site_table_id (str): It holds BigQuery table ID where the results will be stored.
        project_id (str): It holds GCP project ID.
        location (str): It holds location of the BigQuery dataset.
        traffic_yearly_kpis_data_input (Input[Dataset]): It holds the component input dataset 
                                                            containing yearly traffic KPIs DataFrame.
    """
    import pandas as pd
    import pandas_gbq

    df_revenues_per_site = pd.read_parquet(traffic_yearly_kpis_data_input.path)

    df_revenues_per_site["revenue_total_mobile"] = df_revenues_per_site["revenue_voice_mobile"] + \
                                                    df_revenues_per_site["revenue_data_mobile"]

    df_revenues_per_site["opex_clients_commissions_data"] = df_revenues_per_site["revenue_data_mobile"] * \
                                                                opex_comissions_percentage

    df_revenues_per_site["opex_clients_commissions_voice"] = df_revenues_per_site["revenue_voice_mobile"] * \
                                                                opex_comissions_percentage

    df_revenues_per_site["opex_clients_commissions"] = df_revenues_per_site["opex_clients_commissions_data"] + \
                                                        df_revenues_per_site["opex_clients_commissions_voice"]

    df_revenues_per_site = df_revenues_per_site.drop(columns=["week_period"]).drop_duplicates(ignore_index=True)

    pandas_gbq.to_gbq(df_revenues_per_site,
                      revenues_per_site_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists='replace')
