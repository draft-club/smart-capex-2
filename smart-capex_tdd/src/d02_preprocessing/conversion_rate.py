import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.d00_conf.conf import conf
from src.d01_utils.utils import add_logging_info



@add_logging_info
def compute_conversion_rate(predict_yearweek, forecast_data, counter_data, input_template_builder):
    """
    Take data used to create the forecast model and return the forecasted cells and their
    corresponding conversion_rate from the same week one year before
    Rate = total_data_traffic_dl_gb of forecasted data/total_data_traffic_dl_gb of historical data

    Parameters
    ----------
    predict_yearweek: str
        Format YYYYWW, yearweek to take for the forecasted input in RANDim
    forecast_data: pd.DataFrame
        forecasted data
    counter_data: pd.DataFrame
        historical counters data that created the forecast model
    input_template_builder: pd.DataFrame
        RANDim template builder data
    Returns
    -------
    conversion_rate: pd.DataFrame
        Table containing conversion_rate for the cells that have been forecasted
    """
    # Keep only weekly cells that appear in hourly data
    counters_data = counter_data.merge(input_template_builder[["CELL"]], how="inner",
                                       left_on='cell_name', right_on='CELL')
    # Transforming to date format
    counters_data["date"] = pd.to_datetime(counters_data["date"], format="%Y-%m-%d")
    counters_data["week_period"] = (counters_data["date"].dt.isocalendar().year * 100 +
                                    counters_data["date"].dt.isocalendar().week)
    forecast_data["join_col"] = forecast_data['week_period'].apply(lambda x: int(x) - 100)

    # Merge weekly historical and forecasted data with 1 year difference
    # (bring forecastd data minus 1 year)
    conversion_rate = forecast_data.merge(counters_data, how='inner',
                                          left_on=['cell_name', 'join_col'],
                                          right_on=['cell_name', 'week_period'],
                                          suffixes=("_forecasted", "_histo"))
    conversion_rate = conversion_rate.fillna(0)

    # Selection of the week period gived to RANDim
    conversion_rate = conversion_rate[
        conversion_rate["week_period_forecasted"] == int(predict_yearweek)]

    # Calculation of rates on traffics
    conversion_rate["traffic_dl_rate"] = np.where(
        conversion_rate["total_data_traffic_dl_gb_histo"] != 0, np.where(
            conversion_rate["total_data_traffic_dl_gb_forecasted"] != 0,
            conversion_rate["total_data_traffic_dl_gb_forecasted"] /
            conversion_rate["total_data_traffic_dl_gb_histo"], 0),
        conversion_rate["total_data_traffic_dl_gb_forecasted"])
    conversion_rate["traffic_ul_rate"] = np.where(
        conversion_rate["total_data_traffic_ul_gb_histo"] != 0, np.where(
            conversion_rate["total_data_traffic_ul_gb_forecasted"] != 0,
            conversion_rate["total_data_traffic_ul_gb_forecasted"] /
            conversion_rate["total_data_traffic_ul_gb_histo"], 0), 1)

    return conversion_rate[["cell_name", "traffic_dl_rate",
                            "traffic_ul_rate", "week_period_forecasted"]]


@add_logging_info
def change_lte_forecasted(conversion_rate_path, lte_input_path, use_case):
    """
    Take the conversion_rate and the output from randim to transform the network's state into the
    forecasted network's state by applying the rate to the initial value.

    Parameters
    ----------
    conversion_rate_path: str
        path to conversion_rate
    lte_input_path: str
        path to the output from randim in .xlsx
    use_case: str

    Returns
    -------
    lte_forecasted: pd.DataFrame
        Table containing conversion_rate for the cells that have been forecasted
    """
    rate = pd.read_csv(conversion_rate_path, sep='|')
    lte = pd.read_excel(lte_input_path, sheet_name=0, engine='openpyxl')
    lte, header_lte, colnames = pre_process_lte(lte)
    lte.columns = colnames
    improvement = pd.merge(lte[['Cell', conf['COLNAMES']['DLTRAFFICDATA'],
                                conf['COLNAMES']['ULTRAFFICDATA']]],
                           rate[['cell_name', 'traffic_dl_rate', 'traffic_ul_rate']],
                           left_on='Cell', right_on='cell_name')
    improvement[conf['COLNAMES']['DLTRAFFICDATA']] = (
            improvement[conf['COLNAMES']['DLTRAFFICDATA']].astype(float) *
            improvement['traffic_dl_rate'])
    improvement[conf['COLNAMES']['ULTRAFFICDATA']] = (
            improvement[conf['COLNAMES']['ULTRAFFICDATA']].astype(float) *
            improvement['traffic_ul_rate'])
    lte = lte.drop(conf['COLNAMES']['ULTRAFFICDATA'], axis=1)
    lte = lte.drop(conf['COLNAMES']['DLTRAFFICDATA'], axis=1)
    improvement = pd.merge(lte, improvement[['Cell', conf['COLNAMES']['DLTRAFFICDATA'],
                                             conf['COLNAMES']['ULTRAFFICDATA']]], on='Cell')
    improvement = improvement.reindex(columns=colnames)
    improvement.insert(loc=4, column='Mode', value=use_case)
    header_lte.insert(loc=4, column='Mode', value=use_case)
    improvement.columns = header_lte.columns
    lte_forecasted = pd.concat([header_lte, improvement], axis=0)
    return lte_forecasted


def post_process_lte(lte_input_path, lte_output_path):
    """
    Post-processing of lte_all_forecasted to add mode column (FDD/TDD)

    Parameters
    ----------
    lte_input_path: str
        Path of Input
    lte_output_path: str
        Path of Output

    Returns
    -------
    None
    """
    worbook = load_workbook(
        filename=lte_input_path)
    sheet = worbook.active
    sheet['E1'] = ''
    sheet['E2'] = ''
    sheet['E3'] = ''
    sheet['E4'] = ''
    sheet['E5'] = ''
    sheet['E6'] = ''
    sheet['E7'] = ''
    sheet['E8'] = ''
    sheet['E9'] = 'FDD/TDD'
    sheet['E10'] = "Mode"
    worbook.save(lte_output_path)


def pre_process_lte(lte):
    """
    The pre_process_lte function processes an LTE DataFrame to separate its header, column names,
    and data for further analysis.

    Parameters
    ----------
    lte: pd.DataFrame

    Returns
    -------
    lte: pd.DataFrame
         A pandas DataFrame containing the processed LTE data
    header_lte: pd.DataFrame
        A pandas DataFrame containing the header information.
    colnames: object
        An object containing the column names.

    """
    header_lte = lte.iloc[:9]
    colnames = lte.iloc[8]
    lte = lte.iloc[9:]
    lte.columns = colnames
    return lte, header_lte, colnames
