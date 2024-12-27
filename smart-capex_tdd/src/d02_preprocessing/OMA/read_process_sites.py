"""Read Process site"""
import numpy as np
import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import read_file, compute_distance


def sites_preprocessing():
    """
    Take raw data of site and cells and create df_site

    Returns
    -------
    df_processed_sites: pd.DataFrame
    """
    df_sites = read_file(".", conf["PATH"]["RAW_DATA"],
                         'Base de données sites et cellules.xlsx',
                         sheet_name="sites")
    df_sites.drop_duplicates(inplace=True, keep="first")
    df_sites["SiteID"] = df_sites["SiteID"].apply(lambda x: x.upper())
    df_sites = replace_some_sites_names_in_sites_sheet(df_sites)
    df_sites = drop_duplicated_same_sites(df_sites)
    df_sites.reset_index(drop=True, inplace=True)
    df_cells = cells_preprocessing('Base de données sites et cellules.xlsx')
    df_processed_sites = merge_sites_and_cells_dataframe(df_sites, df_cells)
    df_processed_sites.drop(columns=["Latitude_Sector_y", "Longitude_Sector_y"], inplace=True)
    df_processed_sites.rename(columns={
        "SiteID": "site_id",
        "Latitude_Sector_x": "latitude",
        "Longitude_Sector_x": "longitude",
        conf['COLNAMES']['COMMUNE']: "commune",
        conf['COLNAMES']['VILLE']: "ville",
        "Province": "province",
        conf['COLNAMES']['REGION']: "region",
        "SiteName": "site_name",
        "CellName": "cell_name",
        "Sector": "sector",
        "Cell_ID": "cell_id",
        "LAC": "cell_lac",
        "Technology": "cell_tech",
        "HorizBeamwidth": "horizantal_beam_width",
        "VertBeamwidth": "vertical_beam_width",
        "Downtilt": "downtilt",
        "Band": "cell_band",
        "CellName_Azimuth": "cell_name_azimuth",
        "Azimuth": "azimuth",
    }, inplace=True)
    df_processed_sites.drop(
        columns=["BSC_x", "RNC_x", "eNodeBID", "mcc", "mnc", "BSC_y", "RNC_y", "cell_name_azimuth"],
        inplace=True)
    # Chanfe U2100-F1 / U2100-F2 / U2100 -F3 in U2100
    df_processed_sites.replace(["U2100-F3", "U2100-F2", "U2100-F1"], "U2100",
                               inplace=True)
    return df_processed_sites


def replace_some_sites_names_in_sites_sheet(df_sites):
    """
    Replace some site name in some site sheet

    Parameters
    ----------
    df_sites: pd.DataFrame

    Returns
    -------
    df_sites: pd.DataFrame

    """
    # remove line with mormol, MORMOLONE--> MORMOL1, MORMOLTWO--> MORMOL2
    sites_to_be_removed = ["MORMOL"]
    df_sites = df_sites[~df_sites["SiteID"].isin(sites_to_be_removed)]
    df_sites["SiteID"][df_sites["SiteID"] == "MORMOLONE"] = "MORMOL1"
    df_sites["SiteID"][df_sites["SiteID"] == "MORMOLTWO"] = "MORMOL2"
    return df_sites


def drop_duplicated_same_sites(df_sites):
    """
    Drop duplicated sites
    Parameters
    ----------
    df_sites: pd.DataFrame

    Returns
    -------
    df_sites: pd.DataFrame
    """
    df_comparison_between_duplicated_sites = get_df_comparison_between_duplicated_sites(df_sites)
    list_sites_to_drop_only_one = df_comparison_between_duplicated_sites["SiteID"] \
        [df_comparison_between_duplicated_sites["distance"] == 0].to_list()
    indexes_to_drop_iloc = []
    for site in list_sites_to_drop_only_one:
        indexes_to_drop_iloc.append(np.nonzero(df_sites["SiteID"] == site)[0][0])
    indexes_to_drop = [df_sites.iloc[i].name for i in indexes_to_drop_iloc]
    df_sites.drop(index=indexes_to_drop, inplace=True)
    df_sites.reset_index(drop=True)
    return df_sites


def get_df_comparison_between_duplicated_sites(df_sites):
    """
    Get comparison between duplicated sites
    Parameters
    ----------
    df_sites: pd.DataFrame

    Returns
    -------
    df_sites: pd.DataFrame
    """
    list_duplicated_sites = df_sites["SiteID"].value_counts()[ \
        df_sites["SiteID"].value_counts() == 2].index

    list_comparison_between_sites = []
    for site in list_duplicated_sites:
        site1 = df_sites[df_sites["SiteID"] == site].iloc[0]
        site2 = df_sites[df_sites["SiteID"] == site].iloc[1]
        p1 = site1[["Latitude_Sector", "Longitude_Sector"]]
        p2 = site2[["Latitude_Sector", "Longitude_Sector"]]
        lat1, lon1 = p1["Latitude_Sector"], p1["Longitude_Sector"]
        lat2, lon2 = p2["Latitude_Sector"], p2["Longitude_Sector"]
        list_comparison_between_sites.append([site,
                                              round(compute_distance(lat1, lon1, lat2, lon2), 3),
                                              site1["Commune "] == site2["Commune "],
                                              site1["Ville "] == site2["Ville "],
                                              site1["Province"] == site2["Province"],
                                              site1["Région "] == site2["Région "],
                                              site1["BSC"] == site2["BSC"],
                                              site1["RNC"] == site2["RNC"],
                                              ])

    return pd.DataFrame(list_comparison_between_sites, columns=["SiteID", "distance",
                                                                "is_commune_similar",
                                                                "is_ville_similar",
                                                                "is_province_similar",
                                                                "is_region_similar",
                                                                "is_bsc_similar",
                                                                "is_rnc_similar"])


def cells_preprocessing(filename):
    """
    Preprocesses the cells

    Parameters
    ----------
    filename: str
        The name of the file to preprocess

    Returns
    -------
    df_cells: pd.DataFrame
    """
    df_cells = read_file(".", conf["PATH"]["RAW_DATA"], filename,
                         sheet_name="cellules")
    df_cells.drop_duplicates(inplace=True, keep="first")
    df_cells = add_cell_band(df_cells)
    df_cells.drop(columns=["key_0", "Unnamed: 2", "Sector_y", "Technology_y"], inplace=True)
    df_cells.rename(columns={"Sector_x": "Sector", "Technology_x": "Technology"}, inplace=True)
    df_cells = df_cells[
        df_cells["Technology"] != "2G"]  # df_cells["Technology"].unique() --> ['2G', '3G', '4G']
    df_cells["SiteName"] = df_cells["SiteName"].apply(lambda x: x.upper())
    df_cells["SiteID"] = df_cells["SiteID"].apply(lambda x: x.upper())
    df_cells = create_new_columns_for_cells(df_cells)
    df_cells = replace_some_sites_names_in_cells_sheet(df_cells)  # first run
    # ----------------------------------------------------------
    df_cells.reset_index(drop=True, inplace=True)
    df_cells = deal_with_names_ids_differences(df_cells)
    df_cells = replace_some_sites_names_in_cells_sheet(df_cells)  # second run
    return df_cells


def add_cell_band(df_cells):
    """
    Read file of Band Network and merge with df_cells

    Parameters
    ----------
    df_cells: pd.DataFrame

    Returns
    -------
    df_cells: pd.DataFrame

    """
    df_band = read_file(".", conf["PATH"]["RAW_DATA"],
                        "deployment_history/Band Network VF.xlsx",
                        sheet_name="Sheet1")
    df_cells["Sector"] = df_cells["Sector"].astype(str)
    df_band = df_band.astype(str)
    return pd.merge(left=df_cells, right=df_band, left_on=df_cells["Sector"],
                    right_on=df_band["Sector"], how='inner', validate='one_to_one')


def deal_with_names_ids_differences(df_cells):
    """
    Work with ids difference on df_cells

    Parameters
    ----------
    df_cells: pd.DataFrame

    Returns
    -------
    df_cells: pd.DataFrame
    """
    # before the merge: some checking site name and site id ::
    # sometimes they are different 122 site in total
    # to decide what to take we will be based on cellname
    df_sitename_diffthan_siteid = df_cells[df_cells["SiteName"].apply(lambda x: x.upper()) \
                                           != df_cells["SiteID"].apply(lambda x: x.upper())][
        ["SiteName",
         "SiteID",
         "CellName"]]
    indexes_that_may_change = df_sitename_diffthan_siteid.index
    df_sitename_diffthan_siteid.apply(replace_by_site_name,
                                      axis=1)  # this operation drop the number to 50
    df_sitename_diffthan_siteid.apply(replace_by_site_id,
                                      axis=1)  # this operation drop the number to 0
    df_cells.loc[indexes_that_may_change] = df_sitename_diffthan_siteid
    return df_cells


def replace_by_site_name(row):
    """
    Replace site names with site name

    Parameters
    ----------
    row: row of pandas dataframe

    Returns
    -------
    row: row of pandas dataframe
    """
    if row["CellName"].startswith(row["SiteName"]) and not row["SiteName"].startswith(
            row["SiteID"]):
        row["SiteID"] = row["SiteName"]
    return row


def replace_by_site_id(row):
    """
    Replace site names with site id

    Parameters
    ----------
    row: row of pandas dataframe

    Returns
    -------
    row: row of pandas dataframe
    """
    if row["CellName"].startswith(row["SiteID"]):
        row["SiteName"] = row["SiteID"]
    return row


def replace_some_sites_names_in_cells_sheet(df_cells):
    """
    Replace some specific site name in cells sheet

    Parameters
    ----------
    df_cells: pd.DataFrame

    Returns
    -------
    df_cells: pd.DataFrame
    """
    # remove 'L1800', 'DBS'
    # MORMOLONE--> MORMOL1, MORMOLTWO--> MORMOL2
    # MAG046ONE --> "MAG0461", MAG046TWO -->"MAG0462"
    # COMPT_TAZ200 --> "TAZ200"
    try:
        sites_to_be_removed = ["L1800", "DBS"]
        df_cells = df_cells[~df_cells["SiteID"].isin(sites_to_be_removed)]
        df_cells["SiteID"][df_cells["SiteID"] == "MAG046ONE"] = "MAG0461"
        df_cells["SiteID"][df_cells["SiteID"] == "MAG046TWO"] = "MAG0462"
        df_cells["SiteID"][df_cells["SiteID"] == "MORMOLONE"] = "MORMOL1"
        df_cells["SiteID"][df_cells["SiteID"] == "MORMOLTWO"] = "MORMOL2"
        df_cells["SiteID"][df_cells["SiteID"] == "COMPT_TAZ200"] = "TAZ200"
    except ValueError:
        # second run on df_cells
        df_cells["SiteID"][df_cells["SiteID"] == "MORMOLONE"] = "MORMOL1"
        df_cells["SiteID"][df_cells["SiteID"] == "MORMOLTWO"] = "MORMOL2"
    return df_cells


def create_new_columns_for_cells(df_cells):
    """
    Creates new columns for df_cells

    Parameters
    ----------
    df_cells: pd.DataFrame

    Returns
    -------
    df_cells: pd.DataFrame
    """
    # we have these three cells duplicated ['BEN942W', 'BEN942V', 'BEN942U']
    # in cell name but different azimuth
    # we can either cellname as primary key (but should watch out for these three cells)
    # or the new column here
    col = "CellName_Azimuth"
    df_cells[col] = df_cells["CellName"] + "_" + df_cells["Azimuth"].astype(str)
    return df_cells


def merge_sites_and_cells_dataframe(df_sites, df_cells):
    """
    Merges sites and cells dataset
    Parameters
    ----------
    df_sites: pd.DataFrame
        Site's information
    df_cells: pd.DataFrame
        Cell's information
    Returns
    -------
    df_merged: pd.DataFrame
    """
    df_merged = pd.merge(left=df_sites,
                         right=df_cells,
                         left_on="SiteID",
                         right_on="SiteID",
                         how="inner").reset_index(drop=True)
    df_merged = df_merged.drop_duplicates(keep='first')
    return df_merged
