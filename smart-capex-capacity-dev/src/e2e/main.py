import configparser
import time
from d02_preprocessing.quad_population import Make_quad_population
from d02_preprocessing.quad_coverage import Get_coverage
from d02_preprocessing.cell_users_by_timing_advance import Get_cell_users
from d01_utils.utils import read_file
import warnings
warnings.filterwarnings('ignore')


def main(config, make_quad_population, get_coverage, get_cell_users):
    """
    end 2 end function
    """

    time_init = time.time()

    #source_directory = config["directories"]["source_directory"]
    source_directory = "../"
    data_directory = config["directories"]["data_directory"]

    # For the boundary of the country
    country = "Côte d'Ivoire"
    file_map_ci = config["filenames"]["file_map_ci"]
    file_map_sous_prefecture = config["filenames"]["file_map_sous_prefecture"]
    map_country = read_file(source_directory, data_directory, file_map_ci)
    map_sous_prefecture = read_file(source_directory, data_directory, file_map_sous_prefecture)

    # Generate quad map
    population_country_quad = make_quad_population.get_quaded_map()

    # Add Regions
    population_country_quad = make_quad_population.fill_quad_with_fields(population_country_quad, map_country,
                                                                         ["REGION", "ID_REGION"])
    # Add sous prefectures
    population_country_quad = make_quad_population.fill_quad_with_fields(population_country_quad, map_sous_prefecture,
                                                                         ["ADM3_FR", "ADM3_PCODE"])

    print("Après quaded map : ", round(time.time() - time_init))

    gdf_quaded_map_population_coverage, gdf_merge_site_position_footprint = get_coverage.get_coverage_intersect_quad(
        population_country_quad)
    print("Après Couverture : ", round(time.time() - time_init))


if __name__ == "__main__":
    config_file = "../d10_experimental/config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    make_quad_population = Make_quad_population(config)
    get_coverage = Get_coverage(config)
    get_cell_users = Get_cell_users(config)

    main(config, make_quad_population, get_coverage, get_cell_users)
