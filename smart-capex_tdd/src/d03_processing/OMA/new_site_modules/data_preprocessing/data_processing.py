import pandas as pd
import numpy as np
from geopy import distance

BANDS = ["L1800", "L2600", "L800", "U2100", "U900"]
def _compute_deployment_date(site_densif, traffic, traffic_feature="total_data_traffic_dl_gb"):
    """
    The _compute_deployment_date function calculates the deployment date for each site based on
    traffic data. It reindexes the traffic data to fill in missing dates and determines the
    deployment date as the last week with zero traffic or the first available date.

    Parameters
    ----------
    site_densif: pd.DataFrame
        DataFrame containing site IDs
    traffic: pd.DataFrame
        DataFrame containing traffic data with columns site_id, date, and traffic feature.
    traffic_feature: str
        Optional string specifying the traffic feature column name,
        default is "total_data_traffic_dl_gb"

    Returns
    -------
    A DataFrame with site IDs and their corresponding deployment dates, merged with the original
    site_densif DataFrame.

    """
    deployment_history = {"site_id": [], "deployment_date": []}
    for s in site_densif.site_id.to_list():

        traffic_data = traffic[(traffic.site_id == s)]

        if len(traffic_data) != 0:

            traffic_data_reindex = __traffic_reindex(traffic_data[[traffic_feature, "date"]])

            if len(traffic_data_reindex[traffic_data_reindex[traffic_feature] == 0]) != 0:
                deployment_date = (
                        traffic_data_reindex[traffic_data_reindex[traffic_feature] == 0].tail(
                    1).index.item() + pd.offsets.Week(1, weekday=6))
            else:
                deployment_date = traffic_data_reindex.head(1).index.item()
            deployment_history["site_id"].append(s)
            deployment_history["deployment_date"].append(deployment_date)
    deployment_history = pd.DataFrame(deployment_history)
    deployment_history.index = deployment_history.site_id
    del deployment_history["site_id"]

    return site_densif.join(deployment_history).drop_duplicates().dropna()


def _compute_neighbors(deployment_localizations, sites_dataset, max_deployment_dist, nb_neighbors):
    """
    The _compute_neighbors function identifies neighboring sites within a specified distance from
    deployment locations and groups them based on proximity and technology type. It returns a
    dataset of these groups and a dictionary mapping group IDs to deployment sites.

    Parameters
    ----------
    deployment_localizations: pd.DataFrame
        DataFrame containing deployment site information with columns site_id, latitude, longitude,
        and deployment_date.
    sites_dataset: pd.DataFrame
        DataFrame containing site information with columns site_id, latitude, longitude,
        and cell_tech.
    max_deployment_dist: int
        Maximum distance to consider for neighboring sites.
    nb_neighbors: int
        Number of neighbors to include per technology type.
    Returns
    -------
    group_dataset: pd.DataFrame
        DataFrame containing grouped neighboring sites with columns site, groupid,
    dico_group_deployed_site: dict
        Dictionary mapping group IDs to deployment site IDs.

    """
    group_dataset = {"site": [], "groupid": [], "distance_from_deployment": [],
                     "deployment_date": [], "tech": []}

    def compute_distance(x, loc_deployment):
        return distance.distance(loc_deployment, (x["latitude"], x["longitude"])).km + 1e-4

    groupid = 0
    for r in deployment_localizations.iterrows():
        group_dataset["site"].append(r[1]["site_id"])

        group_dataset["groupid"].append(groupid)
        group_dataset["distance_from_deployment"].append(0)
        group_dataset["tech"].append("both")
        deployment_date = r[1]["deployment_date"]

        loc_deployment = (r[1]["latitude"], r[1]["longitude"])

        sites_dataset["distance"] = sites_dataset.apply(
            compute_distance, axis=1, loc_deployment=loc_deployment
        )

        group = sites_dataset[
            (sites_dataset["distance"] <= max_deployment_dist) & (sites_dataset["distance"] > 0) & (
                    sites_dataset["site_id"] != r[1]["site_id"])].sort_values(
            "distance").groupby("cell_tech").head(nb_neighbors)

        group_dataset["site"] += group['site_id'].to_list()
        group_dataset["groupid"] += [groupid] * len(group)
        group_dataset["tech"] += group['cell_tech'].to_list()
        group_dataset["distance_from_deployment"] += group['distance'].to_list()
        group_dataset["deployment_date"] += [deployment_date] * (len(group) + 1)
        groupid += 1

    group_dataset = pd.DataFrame(group_dataset).dropna()
    print(group_dataset.head())
    dico_group_deployed_site = dict(zip(range(group_dataset.groupid.max() + 1),
                                    deployment_localizations.site_id.to_list()))

    group_dataset.index = (group_dataset.site + group_dataset.tech +
                           group_dataset.groupid.astype('str'))
    return group_dataset, dico_group_deployed_site


def _make_traffic_dataset(group_dataset, traffic_dataset, nb_weeks_after, nb_weeks_before,
                          traffic_feature="total_data_traffic_dl_gb"):
    """
    The _make_traffic_dataset function processes traffic data for different groups and sites,
    reindexes the data to a weekly frequency, and combines it with group information to create a
    comprehensive dataset for analysis.

    Parameters
    ----------
    group_dataset: pd.DataFrame
         DataFrame containing group information with columns groupid, site,tech,and deployment_date.
    traffic_dataset: pd.DataFrame
        DataFrame containing traffic data with columns site_id, cell_tech, date,
        and traffic feature (default is total_data_traffic_dl_gb).
    nb_weeks_after: int
        Number of weeks after the deployment date to consider.
    nb_weeks_before: int
        Number of weeks before the deployment date to consider.
    traffic_feature: str
        The traffic feature to be analyzed (default is total_data_traffic_dl_gb).

    Returns
    -------
    group_traffic_dataset: pd.DataFrame
        A DataFrame containing the reindexed traffic data combined with the group information,
        with rows corresponding to each site and technology combination.

    """
    group_traffic_dataset = []
    group_traffic_dataset_index = []

    for gr, df in group_dataset.groupby("groupid"):

        sites = df.site.to_list()
        techs = df.tech.to_list()
        deployment_date = df.deployment_date.to_list()[0]
        min_date = deployment_date.start_time - pd.offsets.Week(nb_weeks_before)
        max_date = deployment_date.start_time + pd.offsets.Week(nb_weeks_after)
        for site, tech in zip(sites, techs):
            if tech == "both":
                traffic_data = traffic_dataset[(traffic_dataset.site_id == site)][
                    ["date", traffic_feature]]
            else:
                traffic_data = traffic_dataset[(traffic_dataset.site_id == site) &
                                               (traffic_dataset.cell_tech == tech)][
                    ["date", traffic_feature]]
            if len(traffic_data) != 0:
                traffic_data_reindex = __traffic_reindex(traffic_data, max_date=max_date,
                                                         min_date=min_date)
                if traffic_data_reindex[traffic_feature].sum() != 0:
                    traffic_data_reindex.index = list(range(-nb_weeks_before, nb_weeks_after + 1,
                                                            1))
                    group_traffic_dataset.append(traffic_data_reindex.T)
                    group_traffic_dataset_index.append(site + tech + str(gr))

    group_traffic_dataset = pd.concat(group_traffic_dataset)

    group_traffic_dataset.index = group_traffic_dataset_index

    group_traffic_dataset = group_traffic_dataset.join(group_dataset).dropna().drop_duplicates()
    return group_traffic_dataset


def _check_valid_groups(group_dataset, dico_group_deployed_site):
    """
    The _check_valid_groups function filters out invalid groups from a dataset based on specific
    criteria: group size, site deployment, and distance from deployment.

    Parameters
    ----------
    group_dataset: pandas.DataFrame
        A DataFrame containing group data with columns such as groupid, site, and
    dico_group_deployed_site: dict
        A dictionary mapping group IDs to their respective deployed sites.

    Returns
    -------
    cleaned_group_dataset: pd.DataFrame
         A DataFrame containing only the valid groups.
    """
    to_remove = []
    for gr, df in group_dataset.groupby("groupid"):
        if len(df) < 2:
            to_remove.append(gr)

        if dico_group_deployed_site[gr] not in df.site.to_list():
            to_remove.append(gr)

        if sum(df.distance_from_deployment.to_list()) == 0:
            to_remove.append(gr)

    cleaned_group_dataset = group_dataset[~group_dataset.groupid.isin(set(to_remove))]

    return cleaned_group_dataset


def _prepare_final_dataset_train(group_traffic_dataset):
    """
    The _prepare_final_dataset_train function filters and processes a dataset to create a final
    dataset containing specific groups with non-zero target traffic.
    It calculates the mean traffic for certain columns and retains only relevant columns.

    Parameters
    ----------
    group_traffic_dataset: pd.DataFrame
        A pandas DataFrame containing traffic data with columns for group IDs, distances,
        and traffic measurements.

    Returns
    -------
    final_dataset: pd.DataFrame
        A pandas DataFrame containing groupid, site, and target_traffic for groups with non-zero
        target traffic.

    """
    final_dataset = group_traffic_dataset[group_traffic_dataset.distance_from_deployment == 0]
    final_dataset.index = final_dataset.groupid
    final_dataset['target_traffic'] = (
        final_dataset.loc[:, [16, 17, 18, 19, 20]].apply(np.mean, axis=1))

    final_dataset = final_dataset[["groupid", "site", "target_traffic"]]
    final_dataset = final_dataset[(final_dataset.target_traffic != 0)]

    return final_dataset


def _prepare_final_dataset_pred(group_dataset):
    """
    The _prepare_final_dataset_pred function filters a dataset to include only rows where the
    distance_from_deployment is 0, sets the index to groupid, and selects only the groupid and site
    columns.

    Parameters
    ----------
    group_dataset: pd.DataFrame
        A pandas DataFrame containing columns groupid, site, and distance_from_deployment.

    Returns
    -------
    final_dataset: pd.DataFrame
        A pandas DataFrame with groupid as the index and columns groupid and site, containing only
        rows where distance_from_deployment is 0.

    """
    final_dataset = group_dataset[group_dataset.distance_from_deployment == 0]
    final_dataset.index = final_dataset.groupid

    final_dataset = final_dataset[["groupid", "site"]]

    return final_dataset


def _compute_neighbor_features(group_traffic_dataset):
    """
    The _compute_neighbor_features function calculates various traffic-related features for groups
    of sites based on their distance from a deployment point. It computes both weighted and
    unweighted traffic metrics, as well as technology-specific traffic data for 3G and 4G networks.

    Parameters
    ----------
    group_traffic_dataset: pd.DataFrame
        A pandas DataFrame containing traffic data for different groups of sites, including columns
        for groupid, distance_from_deployment, tech, and traffic metrics for different time periods.

    Returns
    -------
    neighbor_features: pd.DataFrame
        A pandas DataFrame containing aggregated and weighted traffic metrics,
        as well as technology-specific traffic data for each group.

    """
    neighbors_dataset = group_traffic_dataset[(group_traffic_dataset.distance_from_deployment != 0)]
    dist_sum = neighbors_dataset[["groupid", "distance_from_deployment"]].groupby('groupid').sum()
    neighbors_dataset = neighbors_dataset.join(dist_sum, on="groupid", how="left", rsuffix="_sum")

    for c in [-8, -7, -6, -5, -4, -3, -2, -1]:
        neighbors_dataset[str(c) + "_w"] = neighbors_dataset[c] * (1 - (
                neighbors_dataset["distance_from_deployment"] /
                neighbors_dataset["distance_from_deployment_sum"]))

    agg_traffic = neighbors_dataset[[
        "groupid", -8, -7, -6, -5, -4, -3, -2, -1]].groupby("groupid").sum()
    agg_traffic_w = (neighbors_dataset[
        ["groupid", "-8_w", "-7_w", "-6_w", "-5_w", "-4_w", "-3_w", "-2_w", "-1_w"]].
                     groupby("groupid").sum())
    agg_traffic["neighbor_traffic"] = agg_traffic.apply(np.mean, axis=1)
    agg_traffic["neighbor_traffic_weighted"] = agg_traffic_w.apply(np.mean, axis=1)

    tech_traffic = {"groupid": [], "min_dist": [], "mean_dist": [], "traffic_3g": [],
                    "traffic_4g": [], "nb_sites": []}

    for gr, df in neighbors_dataset.groupby("groupid"):

        tech_traffic["groupid"].append(gr)
        min_dist = min(df.distance_from_deployment)
        mean_dist = df.distance_from_deployment.mean()
        tech_traffic['min_dist'].append(min_dist)
        tech_traffic['mean_dist'].append(mean_dist)
        tech_traffic['nb_sites'].append(len(df))

        if "3G" in df.tech.to_list():

            min_dist_3g = min(df[df.tech == "3G"].distance_from_deployment)
            tech_traffic['traffic_3g'].append(np.mean(
                df[(df.distance_from_deployment == min_dist_3g) & (df.tech == "3G")][
                    [-8, -7, -6, -5, -4, -3, -2, -1]].sum(axis=0)))

        else:
            tech_traffic['traffic_3g'].append(0)

        if "4G" in df.tech.to_list():
            min_dist_4g = min(df[df.tech == "4G"].distance_from_deployment)
            tech_traffic['traffic_4g'].append(np.mean(
                df[(df.distance_from_deployment == min_dist_4g) & (df.tech == "4G")][
                    [-8, -7, -6, -5, -4, -3, -2, -1]].sum(axis=0)))

        else:
            tech_traffic['traffic_4g'].append(0)

    tech_traffic = pd.DataFrame(tech_traffic)
    tech_traffic.index = tech_traffic.groupid
    del tech_traffic["groupid"]
    neighbor_features = agg_traffic.join(tech_traffic)
    neighbor_features = neighbor_features.drop([-8, -7, -6, -5, -4, -3, -2, -1], axis=1)
    return neighbor_features


def _compute_target_site_bands(group_traffic_dataset, traffic_dataset):
    """
    The _compute_target_site_bands function calculates the number of active bands for target sites
    within a specified date range and returns this information in a DataFrame. It processes traffic
    data to determine which bands are active based on data traffic and groups the results by site
    and group ID.

    Parameters
    ----------
    group_traffic_dataset: pd.DataFrame
        DataFrame containing columns groupid, site, deployment_date, and distance_from_deployment.
    traffic_dataset: pd.DataFrame
        DataFrame containing columns site_id, cell_name, cell_band, date, and
        total_data_traffic_dl_gb.

    Returns
    -------
    bands: pd.DataFrame
        A DataFrame with the count of active bands for each target site, indexed by groupid.
    """
    target_sites = group_traffic_dataset[group_traffic_dataset.distance_from_deployment == 0][
        ["groupid", "site", "deployment_date"]]

    band_nb = {k + "_target": [] for k in BANDS}
    band_nb['groupid'] = []

    for s, d, gid in zip(target_sites.site, target_sites.deployment_date, target_sites.groupid):

        traffic_data = traffic_dataset[traffic_dataset.site_id == s]

        cell_name_band = dict(
            zip(traffic_data[["cell_name", "cell_band"]].drop_duplicates().cell_name,
                traffic_data[["cell_name", "cell_band"]].drop_duplicates().cell_band))

        min_date = d.start_time + pd.offsets.Week(16)
        max_date = d.start_time + pd.offsets.Week(20)

        traffic_data = traffic_data.groupby(['cell_name', 'date']).sum()

        traffic_data = traffic_data.reset_index().pivot_table(index='date', columns='cell_name',
                                                              values=['total_data_traffic_dl_gb'],
                                                              aggfunc="sum",
                                                              fill_value=0)
        idx = pd.period_range(min_date, max_date, freq="W")
        traffic_data_reindex = traffic_data.reindex(idx, fill_value=0)

        band_nb_site = {k: 0 for k in BANDS}
        for key in cell_name_band:
            if traffic_data_reindex["total_data_traffic_dl_gb", key].sum() != 0:
                band_nb_site[cell_name_band[key]] += 1

        for k in band_nb_site.keys():
            band_nb[k + "_target"].append(band_nb_site[k])
        band_nb["groupid"].append(gid)

    bands = pd.DataFrame(band_nb)
    bands.index = bands.groupid
    del bands["groupid"]

    return bands


def _compute_neighbor_bands(group_traffic_dataset, traffic_dataset):
    """
    The _compute_neighbor_bands function calculates the number of neighboring cells and their
    traffic for specified frequency bands over a given period. It processes traffic data grouped by
    groupid and computes the sum of traffic for each band, returning a DataFrame with the results.

    Parameters
    ----------
    group_traffic_dataset: pd.DataFrame
        DataFrame containing groupid, site, deployment_date, and distance_from_deployment.
    traffic_dataset: pd.DataFrame
        DataFrame containing site_id, cell_name, cell_band, date, and total_data_traffic_dl_gb.

    Returns
    -------
    A DataFrame with the number of neighboring cells and their traffic for each band,
    indexed by groupid.

    """
    group_traffic_dataset = (
        group_traffic_dataset)[group_traffic_dataset.distance_from_deployment != 0][
        ["groupid", "site", "deployment_date"]]

    band_nb = {k + "_neighbor": [] for k in BANDS}
    band_traffic = {k + "_traffic": [] for k in BANDS}
    band_nb["groupid"] = []
    band_traffic["groupid"] = []

    for gr, df in group_traffic_dataset.groupby("groupid"):
        d = df.deployment_date.to_list()[0]
        traffic_data = traffic_dataset[(traffic_dataset.site_id.isin(df.site))]

        cell_name_band = dict(
            zip(traffic_data[["cell_name", "cell_band"]].drop_duplicates().cell_name,
                traffic_data[["cell_name", "cell_band"]].drop_duplicates().cell_band))

        min_date = d.start_time - pd.offsets.Week(8)
        max_date = d.start_time - pd.offsets.Week(1)

        traffic_data = traffic_data.groupby(['cell_name', 'date']).sum()

        traffic_data = traffic_data.reset_index().pivot_table(index='date', columns='cell_name',
                                                              values=['total_data_traffic_dl_gb'],
                                                              aggfunc="sum",
                                                              fill_value=0)
        idx = pd.period_range(min_date, max_date, freq="W")
        traffic_data_reindex = traffic_data.reindex(idx, fill_value=0)

        band_nb_site = {k: 0 for k in BANDS}
        band_traffic_site = {k: 0 for k in BANDS}
        for key in cell_name_band:
            if traffic_data_reindex["total_data_traffic_dl_gb", key].sum() != 0:
                band_nb_site[cell_name_band[key]] += 1
                band_traffic_site[cell_name_band[key]] += traffic_data_reindex[
                                                        "total_data_traffic_dl_gb", key].sum() / 8
        for k in band_nb_site.keys():
            band_nb[k + "_neighbor"].append(band_nb_site[k])
            band_traffic[k + "_traffic"].append(band_traffic_site[k])
        band_nb["groupid"].append(gr)
        band_traffic["groupid"].append(gr)

    bands = pd.DataFrame(band_nb)
    bands.index = bands.groupid

    bands_traffic = pd.DataFrame(band_traffic)
    bands_traffic.index = bands.groupid
    del bands['groupid']
    del bands_traffic["groupid"]

    return bands.join(bands_traffic)


def __add_bands(dataset, bands_nb):
    """
    The __add_bands function adds new columns to a dataset, each corresponding to a specific band
    from the predefined BANDS list. Each new column is named by appending "_target" to the band name
    and is filled with a repeated value from the bands_nb list.

    Parameters
    ----------
    dataset: dict
        a dictionary representing the dataset to be updated.
    bands_nb: list
        a list of numbers corresponding to each band in BANDS.
    Returns
    -------
    dataset: pd.DataFrame
        The function returns the updated dataset dictionary with new columns added.
    """
    for bands, b in zip(BANDS, bands_nb):
        dataset[bands + "_target"] = [b] * len(dataset)

    return dataset


def __traffic_reindex(traffic_data, max_date=None, min_date=None):
    """
    The __traffic_reindex function reindexes traffic data to a weekly frequency, filling in any
    missing dates with zero traffic values. It ensures that the traffic data spans from the minimum
    to the maximum date provided or found in the data.

    Parameters
    ----------
    traffic_data: pd.DataFrame
        A DataFrame containing traffic data with at least 'date' and traffic feature columns.
    max_date: str
        Optional; the maximum date to consider for reindexing.
    min_date:str
        Optional; the minimum date to consider for reindexing.

    Returns
    -------
    traffic_data_reindex: pd.DataFrame
        A DataFrame reindexed to a weekly frequency with missing dates filled with
        zero traffic values.
    """
    if min_date is None:
        min_date = traffic_data.date.min()
    if max_date is None:
        max_date = traffic_data.date.max()

    agg_date = traffic_data.groupby("date").sum()
    idx = pd.period_range(min_date, max_date, freq="W")
    traffic_data_reindex = agg_date.reindex(idx, fill_value=0)
    return traffic_data_reindex
