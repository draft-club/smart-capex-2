from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_site_margin(project_id: str,
                        location: str,
                        margin_per_site_table_id: str,
                        revenues_per_unit_traffic_data_input: Input[Dataset],
                        margin_per_site_data_output: Output[Dataset]):
    """It computes the yearly margin for voice and data.

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        margin_per_site_table_id (str): It holds the resource name on BigQuery
        revenues_per_unit_traffic_data_input (Input[Dataset]): It holds revenues per unit traffic data 
        margin_per_site_data_output (Output[Dataset]): It holds the computed yearly margin per site
    
    Returns:
        margin_per_site_data_output (Output[Dataset]): It holds the computed yearly margin per site
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    revenues_per_unit_traffic = pd.read_parquet(revenues_per_unit_traffic_data_input.path)

    # Compute the margin for voice and data
    # Need to be upweek_dated to compute the cost per different service
    revenues_per_unit_traffic['site_margin_monthly'] = (revenues_per_unit_traffic['revenue_total_mobile'] -
                                                        revenues_per_unit_traffic['opex_clients_commissions'])

    df_margin_per_site = revenues_per_unit_traffic.groupby(['site_id'])[['site_margin_monthly']].sum().reset_index()

    df_margin_per_site.columns = ['site_id', 'site_margin_yearly']

    print("df_margin_per_site shape: ", df_margin_per_site.shape)
    df_margin_per_site.to_parquet(margin_per_site_data_output.path)

    print("columns", df_margin_per_site.info())

    pandas_gbq.to_gbq(df_margin_per_site, margin_per_site_table_id, project_id=project_id,
                      location=location, if_exists='replace')
