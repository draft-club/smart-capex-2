import datetime as dt
import os
import re

import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import get_band, round_50, add_logging_info


def bh_computation(df: pd.DataFrame, tech: str, isp: str):
    """
    Function compute the busy hour

    Parameters
    ----------
    df: pd.DataFrame
        pandas Dataframe containing data from OSS raw KPI with transformation from preprocess_data
    tech: str
        "4G" or "TDD"
    isp: str
        Internet services provider (NOKIA or HUAWEI)

    Return
    ------
    bh_{tech}: pd.DataFrame
        a Dataframe with the corresponding busy hour format of RANDim's template builder
     """
    if isp == 'NOKIA' and tech == "3G":
        bh_3g = df.loc[df.groupby(['RNC', 'SITE', 'SECTOR', 'FREQUENCY BAND', 'CELL'])[
            'HSDPA_BUFF_WITH_DATA_PER_TTI'].idxmax()]
        return bh_3g
    if isp == 'NOKIA':
        df["active_users"] = (df.SUM_ACTIVE_UE_DATA_DL_PLMN / df.DENOM_ACTIVE_UE_DATA_DL_PLMN) - (
                df.IP_TPUT_TIME_DL_QCI_1 / (3600 * 1000))
        df = df.fillna(0)
        bh_4g = df.sort_values(['active_users'], ascending=False).drop_duplicates(
            subset=['CODE_ELT_ENODEB', 'ENODEB', 'SECTOR', 'CODE_ELT_CELL', 'CELL', 'FREQUENCY'],
            keep="first")
        bh_4g = bh_4g.drop("active_users", axis=1)
        return bh_4g
    if isp == 'HUAWEI':
        df[conf['COLNAMES']['ACTIVEUSERDL2']] = df[conf['COLNAMES']['ACTIVEUSERDL2']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL3']] = df[conf['COLNAMES']['ACTIVEUSERDL3']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL4']] = df[conf['COLNAMES']['ACTIVEUSERDL4']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL5']] = df[conf['COLNAMES']['ACTIVEUSERDL5']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL6']] = df[conf['COLNAMES']['ACTIVEUSERDL6']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL7']] = df[conf['COLNAMES']['ACTIVEUSERDL7']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL8']] = df[conf['COLNAMES']['ACTIVEUSERDL8']].apply(
            lambda x: str(x).replace('NIL', '0'))
        df[conf['COLNAMES']['ACTIVEUSERDL9']] = df[conf['COLNAMES']['ACTIVEUSERDL9']].apply(
            lambda x: str(x).replace('NIL', '0'))

        df[conf['COLNAMES']['ACTIVEUSERDL2']] =df[conf['COLNAMES']['ACTIVEUSERDL2']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL3']] =df[conf['COLNAMES']['ACTIVEUSERDL3']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL4']] =df[conf['COLNAMES']['ACTIVEUSERDL4']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL5']] =df[conf['COLNAMES']['ACTIVEUSERDL5']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL6']] =df[conf['COLNAMES']['ACTIVEUSERDL6']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL7']] =df[conf['COLNAMES']['ACTIVEUSERDL7']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL8']] =df[conf['COLNAMES']['ACTIVEUSERDL8']].astype('float')
        df[conf['COLNAMES']['ACTIVEUSERDL9']] =df[conf['COLNAMES']['ACTIVEUSERDL9']].astype('float')

        df["active_users"] = df["L.TRAFFIC.ACTIVEUSER.DL.QCI.2"] + df[
            "L.TRAFFIC.ACTIVEUSER.DL.QCI.3"] + df["L.TRAFFIC.ACTIVEUSER.DL.QCI.4"] + df[
                                 "L.TRAFFIC.ACTIVEUSER.DL.QCI.5"] + df[
                                 "L.TRAFFIC.ACTIVEUSER.DL.QCI.6"] + df[
                                 "L.TRAFFIC.ACTIVEUSER.DL.QCI.7"] + df[
                                 "L.TRAFFIC.ACTIVEUSER.DL.QCI.8"] + df[
                                 "L.TRAFFIC.ACTIVEUSER.DL.QCI.9"]
        df = df.fillna(0)
        bh_4g = df.sort_values(['active_users'], ascending=False).drop_duplicates(
            subset=['CODE_ELT_ENODEB', 'SECTOR', 'CODE_ELT_CELL', 'CELL', 'FREQUENCY'],
            keep="first")
        bh_4g = bh_4g.drop("active_users", axis=1)
        return bh_4g
    return None


def preprocessing_file(raw_oss: pd.DataFrame, tech: str, isp: str):
    """
    Function that take a dataframe from the OSS raw KPIs and create an output to match the
    input of RANDim's template builder depending on the technology used and with the busy hour
    calculated

    Parameters
    ----------
    raw_oss pd.DataFrame
        pandas Dataframe containing data from OSS raw KPI
    tech: str
        "4G" or "TDD"
    isp: str
     Internet services provider (NOKIA or HUAWEI)

    Return
    ------
    input_builder: pd.DataFrame
        a Dataframe with the corresponding format of RANDim's template builder
     """
    input_builder = None
    if isp == 'NOKIA':
        col_name = []
        for i in raw_oss.columns:
            res = re.sub(r"\(.*\)", "", i)
            col_name.append(res.strip().upper())
        raw_oss.columns = col_name
        raw_oss = raw_oss.fillna(0)
        raw_oss["PERIOD_START_TIME"] = raw_oss["PERIOD_START_TIME"].apply(
            lambda x: dt.datetime.strptime(x, "%m.%d.%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"))
        # Select TDD cells
        raw_oss = raw_oss[raw_oss[conf['COLNAMES']['LNCEL_NAME']].str.contains("_L23")]
        raw_oss["MEAN_PRB_AVAIL_PDSCH"] = raw_oss["MEAN_PRB_AVAIL_PDSCH"].apply(round_50)
        raw_oss["SECTOR"] = raw_oss[conf['COLNAMES']['LNCEL_NAME']].apply(
            lambda x: re.search(r"(L\d+)-(\d)", x).group(2))
        raw_oss["FREQUENCY"] = raw_oss["LNCEL NAME"].apply(get_band) + "00"
        raw_oss["FREQUENCY"] = raw_oss["FREQUENCY"].apply(lambda x: x[1:])
        raw_oss = raw_oss.rename(
            columns={"PERIOD_START_TIME": "HOUR_DATE", "LNBTS NAME": "CODE_ELT_ENODEB",
                     conf['COLNAMES']['LNCEL_NAME']: "CELL"})
        raw_oss["CODE_ELT_CELL"] = raw_oss["CELL"]
        raw_oss["ENODEB"] = raw_oss["CODE_ELT_ENODEB"]
        input_builder_grouped = bh_computation(raw_oss, tech, isp=isp)
        input_builder = input_builder_grouped[conf["PREPROCESSING"]["HEADER_4G"]]

    elif isp == 'HUAWEI':
        col_name = []
        for i in raw_oss.columns:
            res = re.sub(r"\(.*\)", "", i)
            col_name.append(res.strip().upper())
        raw_oss.columns = col_name
        raw_oss = raw_oss.fillna(0)
        raw_oss["L.CHMEAS.PRB.DL.AVAIL"] = raw_oss["L.CHMEAS.PRB.DL.AVAIL"].apply(round_50)
        raw_oss["SECTOR"] = raw_oss["LOCALCELL ID"]
        freq_band = conf['PREPROCESSING']['FREQUENCY_BAND']
        raw_oss["FREQUENCY"] = raw_oss['FREQUENCY BAND'].apply(lambda x: freq_band[str(x)])
        raw_oss = raw_oss.rename(
            columns={"DATE": "HOUR_DATE", "ENODEB NAME": "CODE_ELT_ENODEB", "CELL NAME": "CELL"})
        raw_oss["CODE_ELT_CELL"] = raw_oss["CELL"]
        raw_oss["ENODEB"] = raw_oss["CODE_ELT_ENODEB"]
        input_builder_grouped = bh_computation(raw_oss, tech, isp=isp)

        input_builder = input_builder_grouped[conf["PREPROCESSING"]["HEADER_4G"]]

    return input_builder


@add_logging_info
def preprocessing_file_all(tech: str):
    """
    Function that applies preprocessing_file for every file in the directory.
    It takes path containing hourly data from the OSS raw KPI and save and return a file containing
    the BH computed over every dataset

    Parameters
    ----------
    tech:
        "4G" or "TDD"
    Return
    -------
    raw_oss_bh_computed: pd.DataFrame
        a Dataframe with the corresponding format of RANDim template builder
    """
    data_path = str(os.path.join(conf['PATH']['RAW_DATA'], 'OSS_4G', conf['USE_CASE'])) + str('/')
    data_path_fdd = str(os.path.join(conf['PATH']['RAW_DATA'], 'OSS_4G', "FDD")) + str('/')
    data_path_tdd = str(os.path.join(conf['PATH']['RAW_DATA'], 'OSS_4G', "TDD")) + str('/')

    # Split the hourly file if they dont exist
    if not (os.path.isfile(data_path_fdd + str('FDD_raw_oss.csv')) and os.path.isfile(
            data_path_tdd + str('TDD_raw_oss.csv'))):
        split_hourly_file()

    files_name = os.listdir(data_path)
    files_dict = {}
    for file in files_name:
        raw_oss = pd.read_csv(data_path + file, sep='|', index_col=False)
        files_dict[file] = preprocessing_file(raw_oss, tech, isp=conf['ISP'])
    raw_oss_all = files_dict[files_name[0]]
    if len(files_dict) > 1:
        del files_dict[files_name[0]]
    if conf['ISP'] == 'NOKIA':
        for _, file in files_dict.items():
            raw_oss_all = pd.concat([raw_oss_all, files_dict[file]])
            raw_oss_bh_computed = bh_computation(raw_oss_all, tech, isp='NOKIA')
            raw_oss_bh_computed["TRAFFIC_DL"] = (raw_oss_bh_computed["PDCP_SDU_VOL_DL"] - (
                    raw_oss_bh_computed["IP_TPUT_VOL_DL_QCI_1"] / 8)) / 100000
            raw_oss_bh_computed = raw_oss_bh_computed.sort_values(
                ['ENODEB', 'SECTOR', 'FREQUENCY', 'TRAFFIC_DL'], ascending=False).drop_duplicates(
                subset=['ENODEB', 'SECTOR', 'FREQUENCY'], keep="first")
            raw_oss_bh_computed = raw_oss_bh_computed.drop('TRAFFIC_DL', axis=1)

    elif conf['ISP'] == 'HUAWEI':
        raw_oss_bh_computed = bh_computation(raw_oss_all, tech, isp='HUAWEI')
    return raw_oss_bh_computed


def split_hourly_file():
    """
    Function that will split the RAW OSS HOURLY file per FDD/TDD
    """
    # Read main file
    df_hourly = pd.read_csv(os.path.join(conf["PATH"]["RAW_DATA"], 'OSS_4G',
                                         'OMA_EXTRACT_COMPLET.csv'))

    # Split per technology
    df_fdd = df_hourly[df_hourly['Cell FDD TDD Indication'] == "CELL_FDD"]
    df_tdd = df_hourly[df_hourly['Cell FDD TDD Indication'] == "CELL_TDD"]

    # Write per technology
    df_fdd.to_csv(os.path.join(conf["PATH"]["RAW_DATA"], 'OSS_4G', 'FDD', 'FDD_raw_oss.csv'),
                  sep="|", index=False)
    df_tdd.to_csv(os.path.join(conf["PATH"]["RAW_DATA"], 'OSS_4G', 'TDD', 'TDD_raw_oss.csv'),
                  sep="|", index=False)
