from coverage.preprocessing.site_radio import *

def main(configfile, site_radio):
    """
    main function
    """
    country = "Côte d'Ivoire"
    company = "ORANGE"
    technologies = ["2G", "3G"]

    config = configparser.ConfigParser()
    config.read(configfile)
    source_directory = config["directories"]["source_directory"]
    data_directory = config["directories"]["data_directory"]
    file_site_radio = config["filenames"]["file_site_radio"]
    file_dictionary_info = config["filenames"]["file_dictionary_info"]
    grid_side_length = float(config["parameters"]["grid_side_length"])
    columns_to_keep = ['Compagnie de téléphonie', 'Site_ID', 'Cellule_ID',
                       'Technologie (2G,3G, 4G)', 'Département', 'Région',
                       'Latitude (format WSG 84) ', 'Longitude (format WSG 84) ']
    
    # Reading files 
    site_radio_df = site_radio.select_columns(source_directory, data_directory, file_site_radio, columns_to_keep)
    # create grid dataframe
    grid_df = site_radio.create_grid(source_directory, data_directory, file_dictionary_info, country, grid_side_length)
    print("grid is created")
    # add nb of sites and cells in each grid for each technology
    for technology in technologies:
        print(technology, " is added")
        grid_df = site_radio.add_sites_cells_feature(site_radio_df, grid_df, company, technology)
    site_radio.save_df(grid_df, "coverage/preprocessing/grid_df.pkl")

if __name__ == "__main__":
    configfile = "config.ini"
    site_radio = SiteRadio()
    main(configfile, site_radio)
