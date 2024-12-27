from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


 # pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def prepare_train(traffic_kpi_3g: list,
                  traffic_kpi_4g: list,
                  traffic_weekly_kpis_data_input: Input[Dataset],
                  traffic_weekly_kpis_data_output: Output[Dataset]):
    """It is used for merging the traffic KPIs of 3G and 4G with the preparation of the data to be 
    used for prophet

    Args:
        traffic_kpi_3g (list): It holds a list of traffic KPIs of 3G
        traffic_kpi_4g (list): It holds a list of traffic KPIs of 4G
        traffic_weekly_kpis_data_input (Input[Dataset]): It holds the processed traffic weekly KPIs
        traffic_weekly_kpis_data_output (Output[Dataset]): It holds the traffic weekly KPIs data after melting 
                                                            the columns of those KPIs
    """

    # imports
    import numpy as np
    import pandas as pd

    df_traffic_weekly_kpis = pd.read_parquet(traffic_weekly_kpis_data_input.path)
    print("df_traffic_weekly_kpis shape before: ", df_traffic_weekly_kpis.shape)

    # Compute lag that will be useful in the case of cross_validation
    df_traffic_weekly_kpis.loc[:, 'date'] = pd.to_datetime(df_traffic_weekly_kpis['date'], format="%Y-%m-%d")

    min_date = df_traffic_weekly_kpis["date"].min()
    date_column = pd.to_datetime(df_traffic_weekly_kpis["date"])
    df_traffic_weekly_kpis.loc[:, "lag"] = ((date_column - min_date) // pd.Timedelta(1, 'W')) + 1

    # Prepare data to use it in prophet: long data format
    id_vars = ["cell_name", "date", "cell_tech", "cell_band", "site_id", "lag"]
    df = pd.melt(df_traffic_weekly_kpis, id_vars=id_vars, value_vars=np.unique(traffic_kpi_3g + traffic_kpi_4g).tolist())
    df.columns = id_vars + ["traffic_kpis", "traffic_kpi_values"]

    kpis_to_keep = np.unique(traffic_kpi_3g + traffic_kpi_4g).tolist()
    df = df[df["traffic_kpis"].isin(kpis_to_keep)]
    df["traffic_kpi_values"] = df.groupby(
                                ['traffic_kpis', "cell_name"])["traffic_kpi_values"].transform(lambda x: x.fillna(x.mean()))
    df_traffic_weekly_kpis = df.fillna(0)

    print("df_traffic_weekly_kpis shape after: ", df_traffic_weekly_kpis.shape)

    df_traffic_weekly_kpis.to_parquet(traffic_weekly_kpis_data_output.path)
