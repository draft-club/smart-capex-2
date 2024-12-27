# -*- coding: utf-8 -*-
"""
@author: mahmoud
"""

import os
import pickle
import requests
import shapely

import numpy as np
import pandas as pd
import geopandas as gpd

class PolygonsConstructer():

    def __init__(self):
        pass

    def __str__(self):
        return "A class that construct polygons from sous prefecture names"

    def get_list_sps(self, filename):
        """
        get list of sous prefecture from the dataset

        Parameters
        ----------
        filename: str, path to the menage dataset
        get the sheet "Ménages (3)" because it contains "Taille de ménage"
        information

        Returns
        -------
        list_sps: list of strings, list of sous prefectures presented in the dataset
        """
        menage_df = pd.read_excel(filename, sheet_name="Ménages (3)", header=1)
        list_sps = list(menage_df["Sous_Prefecture"].unique())
        return list_sps


    def get_json(self, url):
        """
        Read url and get json from it
        (If you have a proxy problem launch it using google collab)

        Parameters
        ----------
        url: str, full url of a specific sous prefecture

        Returns
        -------
        data: dictionary, information about the specific sous prefecture.
        It contains bboxs, centroid latitude and longitude, ..
        """
        data = requests.get(url).json()
        return data


    def construct_bbox_dict(self, filename, url_start, url_end, country):
        """
        Construct a dictionary containing bbox information extracted from
        Nominatim API about each sous prefecture

        Parameters
        ----------
        filename: str, path to "menage" dataset
        url_start: str, url start "https://nominatim.openstreetmap.org/search.php?q="
        url_end: str, "&polygon_geojson=1&format=json"
        country: str, country name (here côte d'Ivoire)

        Returns
        -------
        dic: dictionary, dictionary of information for each sous prefecture
        """
        dic = {}
        list_sps = self.get_list_sps(filename)
        for sp in list_sps:
            sp_link = sp + "+" + country
            url = os.path.join(url_start, sp_link, url_end)
            dic[sp] = self.get_json(url)
        return dic


    def save_bbox_dict(self, dic):
        """
        Save dictionary containing information about sous/prefectures
        Parameters
        ----------
        dic: dictionary, dictionary with the scraped information
        """
        dic_name = "cities_info.pkl"
        dic_file = open(dic_name, "wb")
        pickle.dump(dic, dic_file)
        dic_file.close()


    def get_bboxes(self, dic_name):
        """
        Get bounding boxes from dictionary. The bounding boxes in Nominatim API
        are in the form [south latitude, north latitude, west longitude, east longitude].
        When converting it to polygon, it should be in the form [x0,y0,x1,y1] where
        x0, x1: (west longitude, east longitude) and (y0, y1): (south latitude, north latitude)

        Parameters
        ----------
        dic_name: dictionary, dictionary already saved
        using save_bbox_dict() and containing the necessary information

        Returns
        -------
        sous_prefectures_df: gpd.GeoDataFrame,
        """
        sous_prefectures_bbox = []
        data = pd.read_pickle(dic_name)
        cols = ["south lat", "north lat", "west lon", "east lon"]
        sous_prefectures = list(data.keys())
        for sous_prefecture in sous_prefectures:
            try:
                sous_prefecture_info = data[sous_prefecture]
                list_row_df = []
                for i in range(len(sous_prefecture_info)):
                    row = np.array([float(e) for e in sous_prefecture_info[i]["boundingbox"]]).reshape(1, -1)
                    list_row_df.append(pd.DataFrame(row, columns=cols))
                df = pd.concat(list_row_df)
                mins = df[["south lat", "west lon"]].min(axis=0).values
                maxs = df[["north lat", "east lon"]].max(axis=0).values
                (slat, wlon), (nlat, elon) = mins, maxs
                x0, y0, x1, y1 = wlon, slat, elon, nlat # critical line of code! watch out!
                sous_prefectures_bbox.append([sous_prefecture, shapely.geometry.box(x0, y0, x1, y1)])

            except:
                sous_prefectures_bbox.append([sous_prefecture, np.nan])
                # print ("check error in city {city}".format(city=city))

        sous_prefectures_df = gpd.GeoDataFrame(sous_prefectures_bbox, columns=["Sous_Prefecture", "geometry"])
        return sous_prefectures_df
