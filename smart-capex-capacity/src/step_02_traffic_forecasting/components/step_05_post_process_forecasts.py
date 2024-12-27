from kfp.dsl import Dataset, Input, component
from utils.config import pipeline_config


# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def post_process_forecasts(project_id: str,
                           location: str,
                           exec_time: str,
                           predicted_traffic_kpis_table_id: str,
                           predicted_traffic_kpis_data_input: Input[Dataset]):
    """It is mainly used to save the predicted traffic KPIs into BigQuery

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        exec_time (str): It represents the week_date_run
        predicted_traffic_kpis_table_id (str): It holds the resource name in BigQuery for saving the 
                                                predicted traffic KPIs
        predicted_traffic_kpis_data_input (Input[Dataset]): It holds the clipped predicted traffic KPIs
        predicted_traffic_kpis_data_output (Output[Dataset]): It holds the predicted traffic KPIs after formatting
                                                              and before being saved into database
    """
    # imports
    import pandas as pd
    import pandas_gbq

    df_predicted_traffic_kpis = pd.read_parquet(predicted_traffic_kpis_data_input.path)
    print("df_predicted_traffic_kpis before: ", df_predicted_traffic_kpis.shape)
    print("value_counts: ",  df_predicted_traffic_kpis["traffic_kpis"].value_counts())

    df_predicted_traffic_kpis["week_date_run"] = exec_time
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.rename(columns={'ds': 'date',
                                                                          'yhat': 'value'})

    df_predicted_traffic_kpis = df_predicted_traffic_kpis.drop(
                                                columns=["level_0", "level_1", "level_2", "level_3", "level_4", "level_5"])

    # get traffic plots and errors if cross_validation is true
    df_predicted_traffic_kpis = df_predicted_traffic_kpis.pivot_table(
                                        index=["cell_name", "date", "cell_tech", "cell_band", "site_id", "week_date_run"],
                                        columns='traffic_kpis')['value'].reset_index()

    df_predicted_traffic_kpis["year"] = df_predicted_traffic_kpis["date"].dt.year
    df_predicted_traffic_kpis["week"] = df_predicted_traffic_kpis["date"].dt.isocalendar().week.astype('int64')
    df_predicted_traffic_kpis["week_period"] = (df_predicted_traffic_kpis["date"].dt.year * 100
                                                +
                                                df_predicted_traffic_kpis["date"].dt.isocalendar().week)

    # get all the columns using for the next steps
    columns_order = ['cell_name', 'date', 'cell_tech', 'cell_band', 'site_id',
                     'year', 'week', 'week_period',
                     'total_voice_traffic_kerlands',
                     'total_data_traffic_dl_gb',
                     'average_number_of_users_in_queue',
                     "average_throughput_user_dl"]

    df_predicted_traffic_kpis = df_predicted_traffic_kpis.reindex(columns_order, axis=1)

    df_predicted_traffic_kpis = df_predicted_traffic_kpis.rename(columns={'date': 'week_date'})
    print("df_predicted_traffic_kpis after: ", df_predicted_traffic_kpis.shape)

    pandas_gbq.to_gbq(df_predicted_traffic_kpis,
                      predicted_traffic_kpis_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists="replace")
