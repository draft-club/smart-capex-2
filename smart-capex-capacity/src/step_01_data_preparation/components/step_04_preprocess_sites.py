from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def preprocess_sites(raw_sites_data_input: Input[Dataset],
                     processed_sites_data_output: Output[Dataset]):
    """Preprocess site data by computing distances between duplicated sites and removing the duplicates for each site.

    Args:
        raw_sites_data_input (Input[Dataset]): It holds the input dataset containing raw site data.
        processed_sites_data_output (Output[Dataset]): It holds the output dataset to store the processed site data.

    """

    # Imports
    import numpy as np
    import pandas as pd
    from math import atan2, cos, radians, sin, sqrt

    # Load Data
    df_sites = pd.read_parquet(raw_sites_data_input.path)
    print("df_sites before columns", df_sites.columns)
    print("df_sites before shape", df_sites.shape)

    # Drop any duplicates (the data was originally loaded from the latest week date)
    df_sites.drop_duplicates(inplace=True, keep="first")

    def compute_distance(latitude_1, longitude_1, latitude_2, longitude_2):
        """Computes the distance between two geographical points.

        Args:
            latitude_1 (float): It holds the latitude of the first point.
            longitude_1 (float): It holds the longitude of the first point.
            latitude_2 (float): It holds the latitude of the second point.
            longitude_2 (float): It holds the longitude of the second point.

        Returns:
            float: The distance between the two points in kilometers.
        """
        radius = 6371.0  # radius of earth in kilometers
        latitude_1, longitude_1, latitude_2, longitude_2 = map(radians, [latitude_1, longitude_1, latitude_2, longitude_2])
        distance_latitude = latitude_2 - latitude_1
        distance_longitude = longitude_2 - longitude_1

        # haversine formula
        a = sin(distance_latitude / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(distance_longitude / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius * c
        return distance

    def get_comparison_between_duplicated_sites(df_sites):
        """Compare duplicated sites to check for similarities in commune and department.

        Args:
            df_sites (pd.DataFrame): It holds the DataFrame containing site data.

        Returns:
            pd.DataFrame: The DataFrame containing comparison results between duplicated sites.
        """
        list_duplicated_sites = df_sites["site_id"].value_counts()[df_sites["site_id"].value_counts() == 2].index

        list_comparison_between_sites = []
        for site in list_duplicated_sites:
            site_1 = df_sites[df_sites["site_id"] == site].iloc[0]
            site_2 = df_sites[df_sites["site_id"] == site].iloc[1]
            point_1 = site_1[["latitude", "longitude"]]
            point_2 = site_2[["latitude", "longitude"]]
            latitude_1, longitude_1 = point_1["latitude"], point_1["longitude"]
            latitude_2, longitude_2 = point_2["latitude"], point_2["longitude"]

            # replaced `ville` with `department`
            list_comparison_between_sites.append([site, round(compute_distance(latitude_1, longitude_1,
                                                                               latitude_2, longitude_2), 3),
                                                  site_1["commune"] == site_2["commune"],
                                                  site_1["department"] == site_2["department"]])


        df_comparison_between_duplicated_sites = pd.DataFrame(list_comparison_between_sites,
                                                              columns=["site_id", "distance",
                                                                       "is_commune_similar",
                                                                       "is_department_similar"])
        return df_comparison_between_duplicated_sites

    for column in df_sites.columns:
        df_sites[column] = df_sites[column].apply(lambda x: x.upper() if isinstance(x, str) else x)

    # Function: drop_duplicated_same_sites
    df_comparison_between_duplicated_sites = get_comparison_between_duplicated_sites(df_sites)

    list_sites_to_drop_only_one = df_comparison_between_duplicated_sites["site_id"][
        df_comparison_between_duplicated_sites["distance"] == 0].to_list()

    indexes_to_drop_iloc = []
    for site in list_sites_to_drop_only_one:
        indexes_to_drop_iloc.append(np.where(df_sites["site_id"] == site)[0][0])

    indexes_to_drop = [df_sites.iloc[i].name for i in indexes_to_drop_iloc]

    df_sites.drop(index=indexes_to_drop, inplace=True)
    df_sites = df_sites.reset_index(drop=True)

    print("df_sites after columns", df_sites.columns)
    print("df_sites after shape", df_sites.shape)

    df_sites.to_parquet(processed_sites_data_output.path)
