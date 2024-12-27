from kfp.dsl import (Dataset,
                     Input,
                     Output,
                     component)
from utils.config import pipeline_config


# pylint: disable=C0415
@component(base_image=pipeline_config["base_image"])
def get_affected_cells_with_interactions_between_upgrades(project_id:str,
                                                          location:str,
                                                          cell_affected_table_id:str,
                                                          maximum_weeks_to_group_upgrade:int,
                                                          processed_oss_counter_data_input:Input[Dataset],
                                                          cell_affected_data_output:Output[Dataset]):
    """Processes cell upgrade data to identify cells affected by upgrades and outputs the processed data.
    
    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        cell_affected_table_id (str): It holds the resource name on BigQuery
        maximum_weeks_to_group_upgrade (int): It holds maximum number of weeks to group upgrades
        processed_oss_counter_data_input (Input[Dataset]): It holds the processed OSS counter data
        cell_affected_data_output (Output[Dataset]): It holds the processed cell affected data

    Returns:
        cell_affected_data_output (Output[Dataset]): It holds the processed cell affected data
    """
    import pandas as pd
    import numpy as np
    import pandas_gbq

    df_traffic_weekly_kpis = pd.read_parquet(processed_oss_counter_data_input.path)

    def get_lag_between_two_week_periods(week_period_1, week_period_2):
        """Calculates the lag in weeks between two given week periods.

        Args:
            week_period_1 (int): The first week period
            week_period_2 (int): The second week period

        Returns:
            int: The lag in weeks between the two week periods.
        """
        week_period_1, week_period_2 = str(int(float(week_period_1))), str(int(float(week_period_2)))
        year1 = int(week_period_1[:4])
        week1 = int(week_period_1[-2:])
        year2 = int(week_period_2[:4])
        week2 = int(week_period_2[-2:])
        return - (53 * year1 + week1 - (53 * year2 + week2))


    def correct_sites_with_several_deployment(df_cell_affected):
        """Combines records for sites with several upgrades

        Args:
            df_cell_affected (pd.DataFrame): It holds cells affected by the upgrade

        Returns:
            df_cell_affected (pd.DataFrame): It holds combined upgrade records 
        """
        df_cell_affected["starting_week_site"] = df_cell_affected["starting_week_site"].min()
        number_of_upgrades = df_cell_affected["week_of_the_upgrade"].drop_duplicates().values

        number_of_upgrades = number_of_upgrades.tolist()
        number_of_upgrades.sort()

        for i in range(0, len(number_of_upgrades) - 1):
            if (abs(get_lag_between_two_week_periods(str(number_of_upgrades[i]),
                                                     str(number_of_upgrades[i + 1]))) < maximum_weeks_to_group_upgrade):
                ## Remove iteration between different upgrades if there is less than 4 weeks
                df_week_of_the_upgrade = df_cell_affected[df_cell_affected["week_of_the_upgrade"] == number_of_upgrades[i]]
                bands_to_eliminate = df_week_of_the_upgrade["bands_upgraded"].drop_duplicates()
                bands_to_eliminate = bands_to_eliminate.values[0].split("-")
                bands_to_eliminate = df_cell_affected["cell_band"].isin(bands_to_eliminate)
                week_of_upgrade = df_cell_affected["week_of_the_upgrade"] == number_of_upgrades[i + 1]
                df_cell_affected = df_cell_affected[~(bands_to_eliminate & week_of_upgrade)]

                df_cell_affected["week_of_the_upgrade"] = np.where(
                                                     df_cell_affected["week_of_the_upgrade"] == number_of_upgrades[i],
                                                     number_of_upgrades[i + 1],
                                                     df_cell_affected["week_of_the_upgrade"])

                new_bands = "-".join(df_cell_affected["bands_upgraded"].drop_duplicates())
                new_tech = "-".join(df_cell_affected["tech_upgraded"].drop_duplicates())

                df_cell_affected["tech_upgraded"] = np.where(
                                                    df_cell_affected["week_of_the_upgrade"] == number_of_upgrades[i + 1],
                                                    new_tech,
                                                    df_cell_affected["tech_upgraded"])

                df_cell_affected["bands_upgraded"] = np.where(
                                                    df_cell_affected["week_of_the_upgrade"] == number_of_upgrades[i + 1],
                                                    new_bands,
                                                    df_cell_affected["bands_upgraded"])
                df_cell_affected = df_cell_affected.drop_duplicates()

        return df_cell_affected

    def get_affected_cells(df_traffic_weekly_kpis):
        """Identifies and categorizes cells affected by upgrades based on traffic weekly KPIs

        Args:
            df_traffic_weekly_kpis (pd.DataFrame): It holds the processed oss counters data

        Returns:
            df_cell_affected_by_upgrade (pd.DataFrame): It holds cells affected by the upgrade
            df_cells_upgraded_sites (pd.DataFrame): It holds cells that were upgraded and sites with multiple upgrades
        """

        df_starting_date_site = (df_traffic_weekly_kpis.groupby("site_id")["week_period"].min().reset_index())

        df_starting_date_site.columns = ["site_id", "starting_week_site"]
        df_traffic_weekly_kpis = df_traffic_weekly_kpis.merge(df_starting_date_site, on="site_id", how="left")


        df_affected = (df_traffic_weekly_kpis.groupby("cell_name")["week_period"].min().reset_index())

        df_affected.columns = ["cell_name", "starting_week_cell"]
        df_traffic_weekly_kpis = df_traffic_weekly_kpis.merge(df_affected, on="cell_name", how="left")


        df_traffic_weekly_kpis["is_upgrade"] = df_traffic_weekly_kpis[["starting_week_cell", "starting_week_site"]] \
                                               .apply(lambda x: 1 if x.iloc[1] < x.iloc[0] else 0, axis=1)

        df_cell_upgraded = df_traffic_weekly_kpis.loc[df_traffic_weekly_kpis["is_upgrade"] == 1]
        df_cell_not_upgraded = df_traffic_weekly_kpis.loc[df_traffic_weekly_kpis["is_upgrade"] == 0]

        df_site_upgraded = df_cell_upgraded[["site_id", "starting_week_cell", "cell_band"]].drop_duplicates()
        df_site_upgraded.dropna(subset=["cell_band"], inplace=True)
        df_site_upgraded = (df_site_upgraded.groupby(["site_id", "starting_week_cell"])["cell_band"]
                            .apply("-".join).reset_index())

        df_site_upgraded["is_affected"] = 1
        df_site_upgraded.columns = ["site_id", "week_of_the_upgrade", "bands_upgraded", "is_affected"]

        df_site_upgraded_tech = df_cell_upgraded[["site_id", "starting_week_cell", "cell_tech"]].drop_duplicates()

        df_site_upgraded_tech = (df_site_upgraded_tech.groupby(["site_id", "starting_week_cell"])["cell_tech"] \
                                .apply("-".join).reset_index())

        df_site_upgraded_tech.columns = ["site_id", "week_of_the_upgrade", "tech_upgraded"]

        df_site_upgraded = df_site_upgraded.merge(df_site_upgraded_tech, on=["site_id", "week_of_the_upgrade"], how="left")

        ## Cells that are an upgrade can be affected by another upgrade ->
        # when has been more than 1 upgrade on the site on the year
        number_of_upgrades_by_site = (df_site_upgraded[["site_id", "bands_upgraded"]].drop_duplicates()
                                      .groupby(["site_id"])["bands_upgraded"]
                                      .count()
                                      .reset_index())
        number_of_upgrades_by_site = number_of_upgrades_by_site[number_of_upgrades_by_site["bands_upgraded"] > 1]

        ## Get the interaction between cells that are both upgraded and an upgrade
        df_cells_upgraded_sites = df_cell_upgraded[df_cell_upgraded["site_id"].isin(number_of_upgrades_by_site["site_id"])]

        # Cell affected by an upgrade -> cells not upgrades and sites in the list of sites upgraded
        df_cell_affected_by_upgrade = df_cell_not_upgraded[df_cell_not_upgraded["site_id"].isin(df_site_upgraded["site_id"])]

        ## It will duplicate the cell info for each upgrade
        df_cell_affected_by_upgrade = df_cell_affected_by_upgrade.merge(
            df_site_upgraded, left_on=["site_id"], right_on=["site_id"], how="right")

        return df_cell_affected_by_upgrade, df_cells_upgraded_sites


    df_cell_affected, df_cell_both_affected = get_affected_cells(df_traffic_weekly_kpis)
    while df_cell_both_affected.shape[0] > 0:
        df_cell_both_affected.drop(columns=["starting_week_site", "starting_week_cell", "is_upgrade"], inplace=True)

        df_cell_affected_2, df_cell_both_affected = get_affected_cells(df_cell_both_affected)
        df_cell_affected = pd.concat([df_cell_affected, df_cell_affected_2])

    df_cell_affected = (df_cell_affected.groupby("site_id")
                        .apply(correct_sites_with_several_deployment)
                        .reset_index(drop=True))


    df_cell_affected['week_date'] = df_cell_affected['week_date'].astype('datetime64[ns]')

    schema = [{'name': 'week_date', 'type': 'DATE'}]

    df_cell_affected.to_parquet(cell_affected_data_output.path)

    pandas_gbq.to_gbq(df_cell_affected, cell_affected_table_id, project_id=project_id, location=location,
                      if_exists='replace',table_schema=schema)
