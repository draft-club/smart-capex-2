from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def merge_all_improvement_features(upgraded_to_not_consider:list,
                                   kpi_to_compute_upgrade_effect:str,
                                   upgrades_features_typology_data_input:Input[Dataset] ,
                                   traffic_model_features_data_input:Input[Dataset],
                                   capacity_kpis_features_data_input:Input[Dataset],
                                   traffic_site_tech_data_input:Input[Dataset],
                                   list_of_upgrades_data_input:Input[Dataset],
                                   merged_features_data_output:Output[Dataset]):
    """Merge all improvement features and save the result as a parquet file for vertex pipeline.

    Args:
        upgraded_to_not_consider (list): List of upgrades to not consider.
        kpi_to_compute_upgrade_effect (str): KPI to compute the upgrade effect.
        upgrades_features_typology_data_input (Input[Dataset]): Input dataset containing upgrades features typology.
        traffic_model_features_data_input (Input[Dataset]): Input dataset containing traffic model features.
        capacity_kpis_features_data_input (Input[Dataset]): Input dataset containing capacity KPIs features.
        traffic_site_tech_data_input (Input[Dataset]): Input dataset containing traffic site tech data.
        list_of_upgrades_data_input (Input[Dataset]): Input dataset containing list of upgrades.
        merged_features_data_output (Output[Dataset]): Output dataset to store merged features.

    """

    import pandas as pd

    df_upgrade_typology_features = pd.read_parquet(upgrades_features_typology_data_input.path)
    df_traffic_model_features = pd.read_parquet(traffic_model_features_data_input.path)
    df_capacity_kpis_features = pd.read_parquet(capacity_kpis_features_data_input.path)
    df_list_of_upgrades = pd.read_parquet(list_of_upgrades_data_input.path)
    df_traffic_site_tech = pd.read_parquet(traffic_site_tech_data_input.path)

    df_upgrade_typology_features_merge = df_upgrade_typology_features.merge(df_traffic_model_features,
                                                                            on="cluster_key", how="inner")

    # Correction their
    df_upgrade_typology_features_merge["site_id"] = df_upgrade_typology_features_merge["cluster_key"].apply(lambda x:
                                                                                                            x.split("_")[0])

    df_upgrade_typology_capacity_features_merged = df_upgrade_typology_features_merge.merge(df_capacity_kpis_features,
                                                                                            on="site_id", how="left")

    df_list_of_upgrades = df_list_of_upgrades[["cluster_key", "tech_upgraded", "bands_upgraded"]].drop_duplicates()

    df_upgrade_typology_capacity_features_list_upgrades_merged = df_upgrade_typology_capacity_features_merged.merge(
                                                                        df_list_of_upgrades, on="cluster_key", how="left")


    df_upgrade_typology_capacity_features_list_upgrades_merged = df_upgrade_typology_capacity_features_list_upgrades_merged[
                ~df_upgrade_typology_capacity_features_list_upgrades_merged["tech_upgraded"].isin(upgraded_to_not_consider)]

    df_traffic_site_tech = df_traffic_site_tech[["cluster_key", "tech_upgraded",
                                                 f"{kpi_to_compute_upgrade_effect}_tech_mean"]]

    df_merged_features = df_upgrade_typology_capacity_features_list_upgrades_merged.merge(df_traffic_site_tech,
                                                                                          on=["cluster_key",
                                                                                              "tech_upgraded"],
                                                                                          how="left")


    print("df_merged_features shape", df_merged_features.shape)

    df_merged_features.to_parquet(merged_features_data_output.path)
