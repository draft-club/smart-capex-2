from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def add_unique_site_features(processed_sites_data_input: Input[Dataset],
                             unique_site_features_data_output: Output[Dataset]):
    """It adds the unique cel_tech and cell_band for each sites

    Args:
        processed_sites_data_input (Input[Dataset]): It holds the sites data
        unique_site_features_data_output (Output[Dataset]): It holds the unique site features (cel_tech and cell_band) data

    Returns:
        unique_site_features_data_output (Output[Dataset]):  It holds the unique site features (cel_tech and cell_band) data
    """""""""
    # Imports
    import pandas as pd

    # Load Data
    df_sites = pd.read_parquet(processed_sites_data_input.path)

    cell_features = ["cell_tech", "cell_band"]
    df_sites = df_sites[["site_id"] + cell_features]

    for cell_feature in cell_features:
        df_sites = df_sites.dropna(subset=cell_feature)
        df_sites = df_sites.reset_index(drop=True)
        df_feature = df_sites.groupby('site_id')[cell_feature].unique().reset_index()
        df_feature = df_feature.rename(columns={cell_feature: f"{cell_feature}_available"})

        df_sites = df_sites.merge(df_feature, on='site_id', how='left')

    df_sites = df_sites.drop_duplicates(subset=["site_id"])
    df_sites = df_sites.drop(columns=cell_features)

    df_sites['cell_tech_available'] = df_sites['cell_tech_available'].str.join('_')

    print("df_sites shape: ", df_sites.shape)
    df_sites.to_parquet(unique_site_features_data_output.path)
