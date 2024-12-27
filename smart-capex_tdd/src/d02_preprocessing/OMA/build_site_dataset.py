import os

import numpy as np
import pandas as pd

from src.d00_conf.conf import conf
from src.d01_utils.utils import read_file, compute_distance

#conf_loader(parse_args().json_file)


def get_df_comparison_between_duplicated_sites(df_sites):
    """
    The get_df_comparison_between_duplicated_sites function identifies duplicated site entries
    in a DataFrame, compares their geographical coordinates and administrative details,
    and returns a DataFrame summarizing these comparisons.

    Parameters
    ----------
    df_sites: pd.DataFrame
        A DataFrame containing site information with columns such as site_id, latitude, longitude,
        commune, ville, and province.

    Returns
    -------
    pd.DataFrame:
        A DataFrame with columns site_id, distance, is_commune_similar, is_ville_similar,
        and is_province_similar, summarizing the comparisons between duplicated site entries
    """
    list_duplicated_sites = df_sites["site_id"].value_counts()[ \
        df_sites["site_id"].value_counts() == 2].index

    list_comparison_between_sites = []
    for site in list_duplicated_sites:
        site1 = df_sites[df_sites["site_id"] == site].iloc[0]
        site2 = df_sites[df_sites["site_id"] == site].iloc[1]
        p1 = site1[["latitude", "longitude"]]
        p2 = site2[["latitude", "longitude"]]
        lat1, lon1 = p1["latitude"], p1["longitude"]
        lat2, lon2 = p2["latitude"], p2["longitude"]
        list_comparison_between_sites.append([site,
                                              round(compute_distance(lat1, lon1, lat2, lon2), 3),
                                              site1["commune"] == site2["commune"],
                                              site1["ville"] == site2["ville"],
                                              site1["province"] == site2["province"],
                                              ])

    return pd.DataFrame(list_comparison_between_sites, columns=["site_id", "distance",
                                                                "is_commune_similar",
                                                                "is_ville_similar",
                                                                "is_province_similar"])


def drop_duplicated_same_sites(df_sites):
    """
    The drop_duplicated_same_sites function identifies and removes duplicated site entries
    from a DataFrame where the geographical coordinates are identical,
    ensuring only one entry per site remains.

    Parameters
    ----------
    df_sites: pd.DataFrame
        A pandas DataFrame containing site information with columns such as site_id, latitude,
        longitude, commune, province, and ville.

    Returns
    -------
    df_sites: pd.DataFrame
        A pandas DataFrame with duplicated site entries removed,
        retaining only one entry per site with identical coordinates.
    """
    df_comparison_between_duplicated_sites = get_df_comparison_between_duplicated_sites(df_sites)
    list_sites_to_drop_only_one = df_comparison_between_duplicated_sites["site_id"] \
        [df_comparison_between_duplicated_sites["distance"] == 0].to_list()
    indexes_to_drop_iloc = []
    for site in list_sites_to_drop_only_one:
        indexes_to_drop_iloc.append(np.nonzero(df_sites["site_id"] == site)[0][0])
    indexes_to_drop = [df_sites.iloc[i].name for i in indexes_to_drop_iloc]
    df_sites.drop(index=indexes_to_drop, inplace=True)
    df_sites.reset_index(drop=True)
    return df_sites


def replace_some_sites_names_in_sites_sheet(df_sites):
    """
    The replace_some_sites_names_in_sites_sheet function processes a DataFrame of site
    information by removing specific site IDs and renaming certain site IDs to standardized names.

    Parameters
    ----------
    df_sites: pd.DataFrame
        A DataFrame containing site information with columns such as site_id, latitude, longitude,
        commune, province, and ville

    Returns
    -------
    df_sites: pd.DataFrame
        A DataFrame with specified site IDs removed and certain site IDs renamed.
    """
    # remove line with mormol, MORMOLONE--> MORMOL1, MORMOLTWO--> MORMOL2
    sites_to_be_removed = ["MORMOL"]
    df_sites = df_sites[~df_sites["site_id"].isin(sites_to_be_removed)]
    df_sites["site_id"][df_sites["site_id"] == "MORMOLONE"] = "MORMOL1"
    df_sites["site_id"][df_sites["site_id"] == "MORMOLTWO"] = "MORMOL2"
    return df_sites


def sites_preprocessing():
    """
    The sites_preprocessing function processes site and cell data by reading,
    cleaning, merging, and enriching it with additional information.
    It ensures data consistency, removes duplicates, and maps site regions.

    Returns
    -------
    df_processed_sites: pd.DataFrame
        A cleaned and enriched DataFrame containing site and cell information
        with additional region data
    """
    df_sites = read_file(".", conf["PATH"]["RAW_DATA"],
                         conf["filenames"]["file_sites_mapping_cities"],
                         "sites")
    df_sites.columns = ["site_id", "latitude", "longitude", "commune", "province", "ville"]
    df_sites.drop_duplicates(inplace=True, keep="first")
    for col in df_sites.columns:
        df_sites[col] = df_sites[col].apply(lambda x: x.upper() if isinstance(x, str) else x)
    df_sites = replace_some_sites_names_in_sites_sheet(df_sites)
    df_sites = drop_duplicated_same_sites(df_sites)
    df_sites.reset_index(drop=True, inplace=True)
    df_cells = cells_preprocessing()
    df_processed_sites = merge_sites_and_cells_dataframe(df_sites, df_cells)
    df_processed_sites.replace(["U2100-F3", "U2100-F2", "U2100-F1"], "U2100", inplace=True)
    df_processed_sites.dropna(how="any", inplace=True)
    df_processed_sites.reset_index(inplace=True, drop=True)
    df_processed_sites = add_site_region(df_processed_sites)
    return df_processed_sites


def add_site_region(df_processed_sites):
    """
    The add_site_region function enriches a DataFrame containing site information by mapping each
    site's province to its corresponding region using a predefined mapping from another dataset.

    Parameters
    ----------
    df_processed_sites: pd.DataFrame
        A DataFrame containing site information, including a province column.

    Returns
    -------
    df_processed_sites: pd.DataFrame
        A DataFrame with an additional region column, mapping each site's province to its
        corresponding region.
    """
    df_region = read_file(".", conf["PATH"]["RAW_DATA"],
                          conf["filenames"]["file_sites_and_cells"],
                          sheet_name="sites")[["Province", "Région "]]
    df_region.columns = ["province", "region"]
    for col in df_region.columns:
        df_region[col] = df_region[col].apply(lambda x: x.upper())
    df_region.drop_duplicates(inplace=True)
    df_region['province'] = df_region['province'].str.replace('É', 'E')
    df_region['province'] = df_region['province'].str.replace('È', 'E')
    df_region['province'] = df_region['province'].str.replace('Â', 'A')
    dict_region = dict(zip(df_region["province"], df_region["region"]))
    df_processed_sites["region"] = df_processed_sites["province"].map(dict_region)
    return df_processed_sites


def cells_preprocessing():
    """
    The cells_preprocessing function reads and processes 3G and 4G cell data from Excel files,
    standardizes column names, merges the data, maps frequency bands to descriptive labels,
    and cleans up unnecessary columns.

    Returns
    -------
    df_cells : pd.DataFrame
        A pandas DataFrame containing processed cell data with columns:
        site_id, cell_name, cell_tech, and cell_band.
    """
    df_3g = pd.read_excel(
        os.path.join(conf["PATH"]["RAW_DATA"], conf["filenames"]["file_engineering_parameters"]),
        engine="openpyxl",
        sheet_name="Engineering_Parameter_3G")
    df_4g = pd.read_excel(
        os.path.join(conf["PATH"]["RAW_DATA"], conf["filenames"]["file_engineering_parameters"]),
        engine="openpyxl",
        sheet_name="Engineering_Parameter_4G")
    df_4g_tdd = pd.read_excel(
        os.path.join(conf["PATH"]["RAW_DATA"], conf["filenames"]["file_engineering_parameters"]),
        engine="openpyxl",
        sheet_name="Engineering_Parameter_4G_TDD")

    df_3g = df_3g[['NodeBName', 'CellName', 'Longitude', 'Latitude', 'DLFrequency']]
    df_3g.columns = ["site_id", "cell_name", "longitude", "latitude", "band_key"]
    df_3g["cell_tech"] = "3G"
    df_4g = df_4g[['eNodeBName', 'Cell Name', 'Longitude', 'Latitude', 'EARFCN']]
    df_4g.columns = ["site_id", "cell_name", "longitude", "latitude", "band_key"]
    df_4g["cell_tech"] = "4G"
    df_4g_tdd = df_4g_tdd[['eNodeBName', 'Cell Name', 'Longitude', 'Latitude', 'EARFCN']]
    df_4g_tdd.columns = ["site_id", "cell_name", "longitude", "latitude", "band_key"]
    df_4g_tdd["cell_tech"] = "4G"

    df_cells = pd.concat([df_3g, df_4g, df_4g_tdd])

    dict_info_band = {1850: "L1800",
                      3250: "L2600",
                      6200: "L800",
                      42825: "L3500",
                      41825: "L3500",
                      10762: "U2100-F1",
                      10737: "U2100-F2",
                      10712: "U2100-F3",
                      3087: "U900"}

    df_cells["cell_band"] = df_cells["band_key"].map(dict_info_band)
    df_cells.drop(columns=["band_key"], inplace=True)
    df_cells["site_id"] = df_cells["site_id"].apply(lambda x: x.upper())
    df_cells["cell_name"] = df_cells["cell_name"].apply(lambda x: x.upper())
    df_cells.drop(columns=["latitude", "longitude"], inplace=True)
    return df_cells


def merge_sites_and_cells_dataframe(df_sites, df_cells):
    """
    The merge_sites_and_cells_dataframe function merges two DataFrames, df_sites and df_cells,
    on the site_id column using an inner join, ensuring that only matching rows from both
    DataFrames are included in the result.

    Parameters
    ----------
    df_sites: pd.DataFrame
        DataFrame containing site information.
    df_cells: pd.DataFrame
        DataFrame containing cell information

    Returns
    -------
    df_merged: pd.DataFrame
        A merged DataFrame containing only the rows where site_id
        matches in both df_sites and df_cells
    """
    df_merged = pd.merge(left=df_sites,
                         right=df_cells,
                         left_on="site_id",
                         right_on="site_id",
                         how="inner").reset_index(drop=True)
    df_merged = df_merged.drop_duplicates(keep='first')
    return df_merged
