# -*- coding: utf-8 -*-
"""
@author: mahmoud
"""
import os
import pickle
import shapely
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


class SiteRadio:

    def __init__(self):
        pass

    def __str__(self):
        return "A class that handles data in the site radio file"

    def read_file(self, source_directory, data_directory, filename):
        """
        Read file and convert it to dataframe

        Parameters
        ----------
        source_directory : str, source directory of the whole project
        data_directory : str, directory that contains datasets
        filename : str, dataset name

        Returns
        -------
        df: pd.DataFrame, initial dataframe
        """
        filepath = os.path.join(source_directory, data_directory, filename)
        if filename.endswith("xlsx"):
            df = pd.read_excel(filepath,header=0)
        if filename.endswith("gdb"):
            df = gpd.read_file(filepath)
        return df



    def select_columns(self, source_directory, data_directory, filename, columns_to_keep):
        """
        function to select columns  and rename columns

        Parameters
        ----------
        source_directory : str, source directory of the whole project
        data_directory : str, directory that contains datasets
        filename : str, dataset name
        columns_to_keep : list of str,

        Returns
        -------
        df: pd.DataFrame, dataframe with selected columns
        """
        df = self.read_file(source_directory, data_directory, filename)
        df = df[columns_to_keep]
        df.columns = ["company", "site_id", "cell_id",
                      "technology", "department", "region",
                      "latitude", "longitude"]
        return df


    def read_dictionary(self,filename):
        """
        Read dictionary containing information about countries like area and bounds
        Parameters
        ----------
        filename: str, filename

        Returns
        -------
        dic: dict, dictionary of information
        """
        with open(filename, "rb") as file:
            dic = pickle.load(file)
        return dic

    def filter_by_value(self, df, column_value, row_value):
        """
        Filter on a specific (column, value) in the dataframe
        e.g.: filter where company==Orange

        Parameters
        ----------
        df : pd.DataFrame, dataframe after first selection
        column_value : str, column on which column filtering is applied
        row_value : str, cell value on which row filtering is applied

        Returns
        -------
        df: pd.DataFrame, filtered dataframe
        """
        return df[df[column_value] == row_value]



    def create_grid(self, source_directory, data_directory, filepath, country, grid_side_length):
        """
        Construct the map of the squared grids
        (by pierre_gelad)

        Parameters:
        ----------
        filepath : str, the path to the dictionary
        country: str, country which we are interested in
        grid_side_length: float, side each of the square grids in the map (in km)

        Return:
        ------
        grid_df: gpd.GeoDataFrame, dataframe containing gpd.geoseries.Geoseries
        describing each of the describing
        """
        grid_cells = []

        dic = self.read_dictionary(filepath)
        area = dic[country]["area"]
        xmin, ymin = dic[country]["xmin"], dic[country]["ymin"]
        xmax, ymax = dic[country]["xmax"], dic[country]["ymax"]

        n_cells = round(np.sqrt(area/(grid_side_length**2)))
        cell_size = (xmax-xmin)/n_cells
        for x0 in np.arange(xmin, xmax, cell_size):
            for y0 in np.arange(ymin, ymax, cell_size):
                x1 = x0+cell_size
                y1 = y0+cell_size
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
        grid_df = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
        return grid_df


    def add_sites_cells_feature(self,
                                site_radio_df,
                                grid_df,
                                company,
                                technology):
        """
        Compute the number of sites of each technology in each grid
        and add it to grid_df dataframe. It checks if a sites is located
        inside a grid and adds it.


        Parameters
        ----------
        site_radio_df: pd.DataFrame, radio sites initial dataframe
        grid_df : gpd.GeoDataFrame, dataframe with grids geometries
        company: str, Orange/MTN/MOOV
        technology : str, 2G/3G/4G

        Returns
        -------
        grid_df: gpd.GeoDataFrame, dataframe with the added features
        """
        site_radio_df = self.filter_by_value(site_radio_df, "company", company)
        company_xg_df = self.filter_by_value(site_radio_df, "technology", technology)
        feature_nb_sites_name = "_".join(["nb_sites", technology, company])
        feature_nb_cells_name = "_".join(["nb_cells", technology, company])

        points = []
        nb_xg_cells = []
        gps_site_id = company_xg_df.groupby("site_id")
        gps_site_id = [(gp[1][["longitude", "latitude"]].values[0],
                        gp[1].shape[0]) for gp in gps_site_id]

        # assert that it contains only one unique value #
        for j, ele in enumerate(gps_site_id):
            try:
                long, lat = float(ele[0][0]), float(ele[0][1])
                nb_cells = ele[1]
            except:
                print("Could not convert string to float")

            point = Point(long,lat)
            points.append(point)
            nb_xg_cells.append(nb_cells)

        list_sites_by_grid = []
        list_cells_by_grid = []
        for j1, polygon in grid_df.iterrows():
            nb_sites = 0
            nb_cells_by_grid = 0
            for j2, point in enumerate(points):
                if point.within(polygon["geometry"]):
                    nb_sites = nb_sites + 1
                    nb_cells_by_grid = nb_cells_by_grid + nb_xg_cells[j2]
            list_sites_by_grid.append(nb_sites)
            list_cells_by_grid.append(nb_cells_by_grid)

        grid_df[feature_nb_sites_name] = list_sites_by_grid
        grid_df[feature_nb_cells_name] = list_cells_by_grid
        return grid_df


    def save_df(self, df, filepath):
        """
        save dataframe

        Parameters:
        df: dataframe, it could be gpd or pd
        filepath: str, filepath
        """
        if type(df) == gpd.geodataframe.GeoDataFrame:
            with open(filepath, 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #df.to_file(filepath)
        if type(df) == pd.DataFrame:
            df.to_pickle(filepath)


    def get_config(self,configfile):
        """
        read config file

        Parameters:
        ----------
        configfile:str, configuration file

        Returns
        -------
        config : ConfigParser object
            configuration file.
        """
        config = configparser.ConfigParser()
        config.read(configfile)
        return config
