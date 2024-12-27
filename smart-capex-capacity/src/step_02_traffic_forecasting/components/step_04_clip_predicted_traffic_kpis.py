from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def clip_predicted_traffic_kpis(predicted_traffic_kpis_data_input: Input[Dataset],
                                predicted_traffic_kpis_data_output: Output[Dataset]):
    """It is used to clip the df_predicted_traffic_kpis to be within range of 0 and 100

    Args:
        predicted_traffic_kpis_data_input (Input[Dataset]): It holds the predicted traffic KPIs output
        predicted_traffic_kpis_data_output (Output[Dataset]): It holds the predicted traffic KPIs output after
                                            being clipped
    """

    # imports
    import pandas as pd

    df_predicted_traffic_kpis = pd.read_parquet(predicted_traffic_kpis_data_input.path)
    print("df_predicted_traffic_kpis before: ", df_predicted_traffic_kpis.shape)
    print("df_predicted_traffic_kpis info: ", df_predicted_traffic_kpis.info())

    is_negative = df_predicted_traffic_kpis["yhat"] < 0
    df_predicted_traffic_kpis.loc[is_negative, "yhat"] = 0

    print("df_predicted_traffic_kpis after: ", df_predicted_traffic_kpis.shape)

    df_predicted_traffic_kpis.to_parquet(predicted_traffic_kpis_data_output.path)
