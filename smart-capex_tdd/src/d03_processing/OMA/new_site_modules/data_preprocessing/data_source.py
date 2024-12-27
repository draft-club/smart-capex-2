import os
import pandas as pd
from geopy import distance

from src.d00_conf.conf import conf

REGION_COLUMNS = ['region_Béni Mellal-Khénifra', 'region_Drâa-Tafilalet',
                  'region_Eddakhla-Oued Eddahab', 'region_Fès-Meknès',
                  'region_Grand Casablanca-Settat', 'region_Guelmim-Oued Noun',
                  'region_Laayoune-Sakia El Hamra', 'region_Marrakech-Safi', 'region_Oriental',
                  'region_Rabat-Salé-Kénitra', 'region_Souss-Massa',
                  'region_Tanger-Tetouan-Al Hoceima']

def get_sites_dataset(filepath):
    """
    The get_sites_dataset function reads a CSV file containing site information, filters and cleans
    the data by selecting specific columns, removing duplicates, and dropping rows with missing
    values.

    Parameters
    ----------
    filepath: str
        The path to the CSV file containing site data.
    Returns
    -------
    sites: pd.DataFrame
        A cleaned pandas DataFrame containing the selected columns without duplicates and missing
        values.

    """
    sites = pd.read_csv(filepath, sep="|")

    sites = (sites[["site_id", "latitude", "longitude", "cell_tech", "region", "cell_band"]].
             drop_duplicates().dropna())

    return sites


def get_densif_sites_dataset(sites_dataset):
    """
    The get_densif_sites_dataset function merges and processes site data from two Excel files and a
    given dataset to create a consolidated DataFrame of unique site information, including site IDs,
    latitudes, and longitudes.

    Parameters
    ----------
    sites_dataset: pd.DataFrame
        A DataFrame containing site information with columns site_id, latitude, and longitude.

    Returns
    -------
    sites_densifs: pd.DataFrame
        A DataFrame containing unique site information with columns site_id, latitude, and longitude
    """
    sites_densif1 = (
        pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'],
        'sites_MES_full.xlsx'), engine='openpyxl'))[["Site", "Latitude", "Longitude"]]
    sites_densif1.columns = ["site_id", "latitude", "longitude"]
    sites_densif_all = pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'],
        'sites_MES.xlsx'), engine='openpyxl')
    sites_densif2 = sites_dataset[sites_dataset.site_id.isin(sites_densif_all.Site)][
        ["site_id", "latitude", "longitude"]]
    sites_densifs = (pd.concat([sites_densif1, sites_densif2]).drop_duplicates().
                     drop_duplicates("site_id").dropna())

    sites_densifs.index = sites_densifs.site_id
    return sites_densifs


def get_traffic_dataset(filepath):
    """
    The get_traffic_dataset function reads a CSV file containing traffic data, processes it by
    cleaning and transforming the data, and returns a DataFrame with additional columns for year and
    week period.

    Parameters
    ----------
    filepath: str
        A string representing the path to the CSV file
    Returns
    -------
    traffic: pd.DataFrame
        A pandas DataFrame containing the cleaned and transformed traffic data.
    """
    traffic = pd.read_csv(filepath, sep="|",
                          usecols=["date", 'total_data_traffic_dl_gb', "site_id", "cell_tech",
                                   "cell_name", "cell_band", "total_voice_traffic_kerlangs"])
    traffic = traffic.dropna(axis=1, how='all')
    traffic["total_voice_traffic_kerlangs"] = traffic["total_voice_traffic_kerlangs"].fillna(0)
    traffic = traffic.dropna()
    traffic["date"] = pd.to_datetime(traffic.date, format="%Y-%m-%d")
    traffic["year"] = traffic["date"].apply(lambda x: x.year)
    traffic.date = traffic.date.dt.to_period(freq="W")
    return traffic


def add_categorical_features(dataset, randim=False, filepath=""):
    """
    The add_categorical_features function enriches a given dataset with categorical features related
    to cell technology and region. It reads site data, processes it to extract relevant features,
    and merges these features with the input dataset.

    Parameters
    ----------
    dataset: pd.DataFrame
        A pandas DataFrame containing site information.
    randim: bool
        A boolean flag indicating whether to use random site data.
    filepath: str
        A string specifying the path to the random site data file (used if randim is True).

    Returns
    -------
    A pandas DataFrame with additional categorical features related to cell technology and region.
    """
    sites_dataset = get_sites_dataset(
        os.path.join(conf['PATH']['RAW_DATA'], conf['FILE_NAMES']['SITES']))
    if not randim:
        sites_dataset = _get_site_densif_categorical_features(sites_dataset)
    else:
        sites_dataset = _get_site_densif_randim_categorical_features(sites_dataset, filepath)
    features = [f for f in sites_dataset.columns if "region" in f] + ['cell_tech_3G',
                                                                      'cell_tech_4G']
    sites_dataset = sites_dataset[sites_dataset.site_id.isin(dataset.site)]
    print(len(sites_dataset))
    sites_dataset.index = sites_dataset.site_id
    dataset.index = dataset.site
    return dataset.join(sites_dataset[features])


def _get_site_densif_categorical_features(sites_dataset):
    """
    The _get_site_densif_categorical_features function processes site data to extract and enrich it
    with categorical features related to cell technology and region. It combines data from multiple
    sources, cleans it, and adds relevant categorical features.

    Parameters
    ----------
    sites_dataset: pd.DataFrame
        A pandas DataFrame containing site information.

    Returns
    -------
    sites_densifs: pd.DataFrame
        A pandas DataFrame with additional categorical features related to cell technology
        and region

    """
    sites_densif1 = pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'], 'sites_MES_full.xlsx'),
                                  engine='openpyxl')[["Site", "Latitude", "Longitude", 'Band']]
    sites_densif1.columns = ["site_id", "latitude", "longitude", "cell_band"]
    sites_densif1["cell_tech"] = pd.Series(["4G"] * len(sites_densif1))

    sites_densif_all = pd.read_excel(os.path.join(conf['PATH']['RAW_DATA'], 'sites_MES.xlsx'),
                                     engine='openpyxl')
    sites_densif2 = sites_dataset[sites_dataset.site_id.isin(sites_densif_all.Site)][
        ["site_id", "latitude", "longitude", "cell_band", "cell_tech"]]
    sites_densifs = (pd.concat([sites_densif1, sites_densif2]).drop_duplicates().
                     drop_duplicates("site_id").dropna())

    sites_densifs.loc[sites_densifs['cell_band'] == 'G900', 'cell_tech'] = "2G"
    del sites_densifs['cell_band']
    sites_densifs = (sites_densifs[["site_id", "latitude", "longitude", "cell_tech"]].
                     drop_duplicates().dropna())

    sites_densifs = (pd.get_dummies(sites_densifs, columns=['cell_tech']).
                     groupby(["site_id", "latitude", "longitude"],
                     as_index=False).sum())

    print(sites_densifs.head())

    regions = []
    sites_dataset = sites_dataset[["latitude", "longitude", "region"]].drop_duplicates().dropna()

    def find_closest_region(lat, long, sites_dataset):
        loc_deployment = (lat, long)
        sites_dataset["distance"] = sites_dataset.apply(
            lambda x: distance.distance(loc_deployment, (x["latitude"], x["longitude"])).km,
            axis=1)
        region = sites_dataset[sites_dataset.distance ==
                               min(sites_dataset.distance)]["region"].unique()[0]
        return region

    for lat, long in zip(sites_densifs.latitude, sites_densifs.longitude):
        region = find_closest_region(lat, long, sites_dataset)
        regions.append(region)

    sites_densifs["region"] = pd.Series(regions)
    sites_densifs = pd.get_dummies(sites_densifs, columns=['region'], dtype=int)
    sites_densifs.index = sites_densifs.site_id

    return sites_densifs



def _get_site_densif_randim_categorical_features(sites_dataset, filepath):
    """
    he _get_site_densif_randim_categorical_features function processes a dataset of site information
    enriching it with categorical features related to cell technology and region. It reads data from
    an Excel file, cleans and processes it, assigns regions based on proximity, and ensures
    all region columns are present in the final dataset.

    Parameters
    ----------
    sites_dataset: pd.DataFrame
        A pandas DataFrame containing site information with columns latitude, longitude, and region.
    filepath: str
        A string specifying the path to the Excel file containing random site data

    Returns
    -------
    randim_ouput_dataset: pd.DataFrame
        A pandas DataFrame with additional categorical features related to cell technology
        and region, indexed by site_id.

    """
    randim_ouput_dataset = pd.read_excel(filepath, skiprows=2, header=None,
                                         names=["Polygon Number", "site_id", "Congested Cells 3G",
                                                "Congested Cells 4G", "Offloading Carriers 3G",
                                                "Offloading Sectors 4G", "Densification Sites 3G",
                                                "Densification Sites 4G", "latitude", "longitude"],
                                         sheet_name=1, engine='openpyxl')

    randim_ouput_dataset = randim_ouput_dataset[["site_id", "latitude", "longitude"]]
    randim_ouput_dataset = (randim_ouput_dataset[["site_id", "latitude", "longitude"]].
                            drop_duplicates().dropna())
    randim_ouput_dataset = randim_ouput_dataset.assign(cell_tech_3G=[0] * len(randim_ouput_dataset),
                                                       cell_tech_4G=[1] * len(randim_ouput_dataset))


    def find_closest_region(lat, long, sites_dataset):
        loc_deployment = (lat, long)
        sites_dataset["distance"] = sites_dataset.apply(
            lambda x: distance.distance(loc_deployment, (x["latitude"], x["longitude"])).km,
            axis=1)
        region = sites_dataset[sites_dataset.distance ==
                               min(sites_dataset.distance)]["region"].unique()[0]
        return region

    regions = []
    sites_dataset = sites_dataset[["latitude", "longitude", "region"]].drop_duplicates().dropna()
    for lat, long in zip(randim_ouput_dataset.latitude, randim_ouput_dataset.longitude):
        region = find_closest_region(lat, long, sites_dataset)
        regions.append(region)

    randim_ouput_dataset["region"] = pd.Series(regions)
    randim_ouput_dataset = pd.get_dummies(randim_ouput_dataset, columns=['region'], dtype=int)
    for r in REGION_COLUMNS:
        if r not in randim_ouput_dataset.columns:
            randim_ouput_dataset[r] = [0] * len(randim_ouput_dataset)
    randim_ouput_dataset.index = randim_ouput_dataset.site_id

    return randim_ouput_dataset


def get_randim_output_dataset(filepath):
    """
    The get_randim_output_dataset function reads an Excel file, extracts specific columns,
    and adds a deployment date column to the dataset.

    Parameters
    ----------
    filepath: str
        The path to the Excel file to be read.

    Returns
    -------
    randim_ouput_dataset: pd.DataFrame
        A pandas DataFrame containing site_id, latitude, longitude, and deployment_date columns.
    """
    randim_ouput_dataset = pd.read_excel(filepath, skiprows=2, header=None,
                                         names=["Polygon Number", "site_id", "Congested Cells 3G",
                                                "Congested Cells 4G", "Offloading Carriers 3G",
                                                "Offloading Sectors 4G", "Densification Sites 3G",
                                                "Densification Sites 4G", "latitude",
                                                "longitude"], sheet_name=1, engine='openpyxl')

    randim_ouput_dataset = randim_ouput_dataset[["site_id", "latitude", "longitude"]]
    randim_ouput_dataset["deployment_date"] = pd.Series([pd.Period("27-06-2024", "W")] *
                                                        len(randim_ouput_dataset))

    return randim_ouput_dataset
