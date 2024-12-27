from src.d00_conf.conf import conf, conf_loader
from src.d01_utils.utils import limit_area

# Manage import according to current country
conf_loader("OCI")
if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
    from src.d01_utils.utils import read_or_execute_decorator
    from src.d02_preprocessing.OCI.get_several_geodata import Get_several_geodata
    from src.d02_preprocessing.OCI.operations_on_geometries import quad_intersect_geovalues_oci, \
        make_difference_between_new_and_existing_cells_coverage, \
        get_cells_simu_by_sp,get_splitted_cells_coverage_diff_between_sp_oci


def get_class_instances():
    """
    function that initiates a Coverage class object

    Returns
    -------
    coverage_object: coverage class instance
    """
    coverage_object = Get_several_geodata()
    return coverage_object



@read_or_execute_decorator
def get_map_pipeline(coverage_object):
    """
    map elementary pipeline function

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance

    Returns
    -------
    gpd.GeoDataFrame, map of the country at sous prefecture level
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_map_sous_pref()


@read_or_execute_decorator
def get_existing_simulations_pipeline(coverage_object,
                                      simulations_dict,
                                      extract_or_full,
                                      boolean_use_processed_simulations_files):
    """
    simulations elementary pipeline function for both new and existing cells.
    We have the possibility to run full cells simulations or an extract or to select
    new or existing cells simulations.

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance
    simulations_dict: dict, simulation specifications dictionary to run (tech, bands)
    extract_or_full: str, "full" for complete cells simulations or "extract" by limiting area in config file
    boolean_use_processed_simulations_files: boolean, use processed files or raw files

    Returns
    -------
    gdf_cells_coverage: gpd.geodataframe, existing cells coverage geometries
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        gdf_cells_coverage = coverage_object.get_coverage_simulations_all_technos(
            simulations_dict, boolean_use_processed_simulations_files, existing_or_new="existing")
        if extract_or_full == "extract":
            limits = conf['parameters']['limits_area']  # to include only for extract
            gdf_cells_coverage = limit_area(gdf_cells_coverage, limits)
        return gdf_cells_coverage


@read_or_execute_decorator
def get_new_simulations_pipeline(coverage_object,
                                 simulations_dict,
                                 extract_or_full,
                                 boolean_use_processed_simulations_files):
    """
    simulations elementary pipeline function for both new and existing cells.
    We have the possibility to run full cells simulations or an extract or to select
    new or existing cells simulations.

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance
    simulations_dict: dict, simulation specifications dictionary to run (tech, bands)
    extract_or_full: str, "full" for complete cells simulations or "extract" by limiting area in config file
    boolean_use_processed_simulations_files: boolean, use processed files or raw files

    Returns
    -------
    gdf_cells_coverage: gpd.geodataframe, new cells coverage geometries
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        gdf_cells_coverage = coverage_object.get_coverage_simulations_all_technos(
            simulations_dict, boolean_use_processed_simulations_files, existing_or_new="new")
        if extract_or_full == "extract":
            limits = conf['parameters']['limits_area']  # to include only for extract
            gdf_cells_coverage = limit_area(gdf_cells_coverage, limits)
        return gdf_cells_coverage


@read_or_execute_decorator
def get_population_country_pipeline(coverage_object, extract_or_full):  # move type_settlements to function arg
    """
    Population settlements elementary pipeline.
    We have the possibility to have all settlements or an extract by limiting area in config file
    and setting extract_or_full parameter to "extract"
    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance
    extract_or_full: str, "full" for complete cells simulations or "extract" by limiting area in config file

    Returns
    -------
    gdf_country_settlements: gpd.DataFrame, country settlements delivered
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        gdf_country_settlements = coverage_object.get_country_settlements(type_settlements="all")
        if extract_or_full == "extract":
            limits = conf['parameters']['limits_area']  # to include only for extract
            gdf_country_settlements = limit_area(gdf_country_settlements, limits)
        return gdf_country_settlements


@read_or_execute_decorator
def get_user_data_pipeline(coverage_object):
    """
    user data elementary pipeline

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance

    Returns
    -------
    pd.DataFrame, user data specific to the country
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_user_data()

def get_simulations_pipeline(coverage_object, existing_or_new):
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        limits = conf['parameters']['limits_area']  # to include only for test
        gdf_cells_coverage = limit_area(coverage_object
                                        .get_coverage_simulations_all_technos(existing_or_new, True), limits)
        return gdf_cells_coverage
    # "[-3.53, -3.48, 6.725, 6.79]"

def get_cells_position_pipeline(coverage_object):  # move type_settlements to function arg
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_existing_cells_position()



@read_or_execute_decorator
def get_market_share_pipeline(coverage_object):
    """
    market share elementary pipeline

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance

    Returns
    -------
    pd.DataFrame, market share
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        df_market_share = coverage_object.get_market_share()
        return df_market_share


@read_or_execute_decorator
def get_data_eco_full_pipeline(coverage_object):
    """
    economical data elementary pipeline
    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance

    Returns
    -------
    pd.DataFrame, dataframe of economical data containing ARPU per
                service along with other variables like region, department, ..
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_economical_data_full()
# TODO
def get_data_eco_full_pipeline(coverage_object):
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_average_arpu_by_sp()


@read_or_execute_decorator
def get_data_eco_by_dept_pipeline(coverage_object):
    """
    economical data per department and sous prefecture pipeline

    Parameters
    ----------
    coverage_object: Coverage.object, coverage class instance

    Returns
    -------
    pd.DataFrame, dataframe of economical data containing ARPU per
                service grouped by department/sous prefecture
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_data_eco_by_sp_and_dept()


# -------------------------------------------- Potential customers pipeline -------------------------
def get_difference_in_coverage_pipeline(gdf_existing_cells_coverage, gdf_new_cells_coverage):
    """
    making the difference between existing and new cells coverage

    Parameters
    ----------
    gdf_existing_cells_coverage: gpd.DataFrame, geopandas dataframe of existing cells coverage
    gdf_new_cells_coverage: gpd.DataFrame, geopandas dataframe of new cells coverage

    Returns
    -------
    gpd.DataFrame, geopandas dataframe with difference between coverage in geometries
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return make_difference_between_new_and_existing_cells_coverage(gdf_existing_cells_coverage,
                                                                       gdf_new_cells_coverage)


def get_splitted_cells_coverage_between_sp(map, gdf_cells_coverage_diff):
    """

    Parameters
    ----------
    map: gpd.DataFrame,
    gdf_cells_coverage_diff: gpd.DataFrame,

    Returns
    -------

    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return get_cells_simu_by_sp(map, gdf_cells_coverage_diff)

# Added while merge
def get_splitted_cells_coverage_diff_between_sp(map, gdf_existing_cells_coverage, gdf_new_cells_coverage,
                                                population_country):
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return get_splitted_cells_coverage_diff_between_sp_oci(map, gdf_existing_cells_coverage, gdf_new_cells_coverage,
                                                               population_country)


def get_potential_customers_under_splitted_coverture(population_country,
                                                     gdf_cells_coverage_diff_by_sp,
                                                     value_to_intersect):
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return quad_intersect_geovalues_oci(population_country,
                                            gdf_cells_coverage_diff_by_sp)


 # -------------------------------------------- NPV computation pipeline  -------------------------
def get_npv_computation_pipeline(coverage_object,
                                 df_population_diff_by_sp,
                                 df_data_eco_rural_sp_dept,
                                 df_market_share,
                                 df_user_data):
    """
    npv computation elementary pipeline
    Parameters
    ----------
    coverage_object
    df_population_diff_by_sp:
    df_data_eco_rural_sp_dept:
    df_market_share:
    df_user_data:

    Returns
    -------
    pd.DataFrame, dataframe with npv
    """
    if conf['COUNTRY'] == '' or conf['COUNTRY'] == 'OCI':
        return coverage_object.get_npv(df_population_diff_by_sp,
                                       df_data_eco_rural_sp_dept,
                                       df_market_share,
                                       df_user_data)
