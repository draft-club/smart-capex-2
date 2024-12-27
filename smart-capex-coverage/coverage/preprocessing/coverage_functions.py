# -*- coding: utf-8 -*-
"""
@author: pierre
"""
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shapely

class Coverage:

    def __init__(self):
        pass

    def __str__(self):
        return "A class that handles data in the population file"

    def open_gdb_file(self, path_gdb):
        if os.path.exists(path_gdb):
            gdb_content = gpd.read_file(path_gdb)
            return gdb_content
        else:
            print("File don't exist")


    def read_file(self, source_directory, data_directory, filename):
        """
        Read file and convert it to dataframe
        (Written by Mahmoud
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


    def create_grided_map(self, map_c, n_cells, print_map=False):
        """
        function to get the gridded map from the original map

        Parameters
        ----------
        map_c : gdf, original map of the country
        n_cells : int, number of cells in the x axis
        print_map : bool, if we want to print the gridded map or not

        Returns
        -------
        gdf: gpd.DataFrame, geodataframe of the grid
        """
        xmin, ymin, xmax, ymax = map_c.total_bounds
        cell_size = (xmax - xmin) / n_cells
        crs = map_c.crs.srs
        grid_cells = []

        for x0 in np.arange(xmin, xmax + cell_size, cell_size):
            for y0 in np.arange(ymin, ymax + cell_size, cell_size):
                x1 = x0 - cell_size
                y1 = y0 + cell_size
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
        map_cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'],
                                    crs=crs)
        map_cell["area_cell"] = map_cell["geometry"].area

        if print_map:
            map_cell.plot(facecolor="none", edgecolor='grey')

        return map_cell

    def pop_cell_creation(self, map_c, map_cell):
        """
        function that fills the grids with population values
        An intersection is made to avoid double count

        Parameters
        ----------
        map_c : gdf, original map of the country
        map_cell : gdf, gridded map

        Returns
        -------
        gdf: gpd.DataFrame, geodataframe gridded map enriched with population data
        """
        map_cc = map_c.copy()
        map_cc["savedSettlementsGeometry"] = map_cc.geometry
        map_cell_agg = gpd.sjoin(map_cell, map_cc, how="left")
        map_cell_agg["IntersectArea"] = map_cell_agg["geometry"].intersection(map_cell_agg["savedSettlementsGeometry"])
        map_cell_agg["IntersectArea_area"] = map_cell_agg["IntersectArea"].area
        map_cell_agg["pop_cell_intersect"] = map_cell_agg["population"] * (map_cell_agg["IntersectArea_area"]/map_cell_agg["area_settlement"])
        map_cell_agg_gb = map_cell_agg.groupby(map_cell_agg.index)['pop_cell_intersect'].sum()
        map_cell["pop_tot"]=map_cell_agg_gb
        return map_cell

    def get_map(self, country):
        # TODO Use cote_d_ivoire\Cote_d_ivoire.shp
        map_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return map_world[map_world.name == country]