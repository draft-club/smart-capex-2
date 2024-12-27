from kfp.dsl import (Dataset, Input, Output, component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def handle_congested_cells(week_of_the_upgrade: str,
                           b32_aggregated_data_input: Input[Dataset],
                           unique_site_features_data_input: Input[Dataset],
                           detected_cell_congestion_data_input: Input[Dataset],
                           unique_congested_cells_data_output: Output[Dataset],
                           congestion_data_output: Output[Dataset]):
    """It selects the bands to upgrade for the congested cells based on the congestion status

    Args:
        week_of_the_upgrade (str): It holds the week of the upgrade
        b32_aggregated_data_input (Input[Dataset]): It holds the calculated compatible terminals ratio column
        unique_site_features_data_input (Input[Dataset]): It holds the unique site features data
        detected_cell_congestion_data_input (Input[Dataset]): It holds dataframet with the congestion column
        unique_congested_cells_data_output (Output[Dataset]): It holds the unique congested cell_name
        congestion_data_output (Output[Dataset]): It holds the congestion data

    Returns:
        congestion_data_output (Output[Dataset]): It holds the congestion data
    """

    # imports
    import pandas as pd

    def process_4g_congestion(row: pd.Series) -> pd.Series:
        """It selectes the bands to upgrade for the 4G congestion

        Args:
            row (pd.Series): It holds the row of the DataFrame

        Returns:
            row: It holds the row of the DataFrame with the upgraded bands
        """

        cell_band = row["cell_band"]
        compatible_terminals_ratio = row["compatible_terminals_ratio"]

        available_bands = row["cell_band_available"]
        bool_l7 = "L700" in available_bands
        bool_l15 = "L1500" in available_bands
        bool_l18 = "L1800" in available_bands
        bool_l21 = "L2100" in available_bands
        bool_l26 = "L2600" in available_bands

        row["tech_upgraded"] = "4G"

        bool_u900 = "U900" in available_bands
        bool_u2100 = "U2100" in available_bands

        if bool_u900:
            row["bands_upgraded"] = "U900"

        if bool_u2100:
            row["bands_upgraded"] = "U2100"


        # First Branch
        if cell_band == "L800":
            if not bool_l18:
                row["bands_upgraded"] = "L1800"
            else:
                if bool_l7:
                    row["bands_upgraded"] = "densification"
                else:
                    if bool_l15:
                        row["bands_upgraded"] = "densification"
                    else:
                        if (compatible_terminals_ratio * 100) > 40:
                            row["bands_upgraded"] = "L1500"
                        else:
                            row["bands_upgraded"] = "densification"

        # Second Branch
        if cell_band == "L1800":
            if bool_l18 and len(available_bands) == 1:
                row["bands_upgraded"] = "L2600"
            else:
                if bool_l18 and bool_l26 and len(available_bands) == 2:
                    row["bands_upgraded"] = "L2100"
                else:
                    if bool_l15:
                        row["bands_upgraded"] = "densification"
                    else:
                        if (compatible_terminals_ratio * 100) > 40:
                            row["bands_upgraded"] = "L1500"
                        else:
                            row["bands_upgraded"] = "densification"

        # Third Branch
        if cell_band == "L2100":
            if bool_l21 and len(available_bands) == 1:
                row["bands_upgraded"] = "L1800"
            else:
                if bool_l18 and bool_l21 and len(available_bands) == 2:
                    row["bands_upgraded"] = "L2600"
                else:
                    if bool_l15:
                        row["bands_upgraded"] = "densification"
                    else:
                        if (compatible_terminals_ratio * 100) > 40:
                            row["bands_upgraded"] = "L1500"
                        else:
                            row["bands_upgraded"] = "densification"

        # Forth Branch
        if cell_band == "L2600":
            if bool_l26 and len(available_bands) == 1:
                row["bands_upgraded"] = "L1800"
            else:
                if bool_l18 and bool_l26 and len(available_bands) == 2:
                    row["bands_upgraded"] = "L2100"
                else:
                    if bool_l21 and bool_l26 and len(available_bands) == 2:
                        row["bands_upgraded"] = "L1800"
                    else:
                        if bool_l15:
                            row["bands_upgraded"] = "densification"
                        else:
                            if (compatible_terminals_ratio * 100) > 40:
                                row["bands_upgraded"] = "L1500"
                            else:
                                row["bands_upgraded"] = "densification"

        # Fifth Branch
        if cell_band == "L700":
            if bool_l15:
                row["bands_upgraded"] = "densification"
            else:
                if (compatible_terminals_ratio * 100) > 40:
                    row["bands_upgraded"] = "L1500"
                else:
                    row["bands_upgraded"] = "densification"

        return row

    def select_upgrade(row: pd.Series) -> pd.Series:
        """It selects the upgrade based on the congestion status of the cell

        Args:
            row (pd.Series): It holds a row of the DataFrame containing congestion status

        Returns:
            row: It holds the processed row based on congestion status.
        """
        if (row["congestion"] == "3G_CONGESTION") or (row["congestion"] == "4G_CONGESTION"):
            return process_4g_congestion(row)

        if row["congestion"] == "NO_CONGESTION":
            return row

    # Load Data
    df_b32_aggregated = pd.read_parquet(b32_aggregated_data_input.path)
    df_sites = pd.read_parquet(unique_site_features_data_input.path)
    df_cells = pd.read_parquet(detected_cell_congestion_data_input.path)

    df_merged_congestion_b32 = pd.DataFrame()
    df_congestion = pd.DataFrame()
    unique_congested_cells = []
    df_unique_congested_cells = pd.DataFrame({"cell_name": []})

    # Get congestion rows only
    df_cells_congestion = df_cells[df_cells["congestion"] != "NO_CONGESTION"]

    # Get the first week_period of each cell_name
    df_cells_congestion_sorted = df_cells_congestion.sort_values(by=["cell_name", "week_period"], ascending=[True, True])
    df_cells_congestion_filtered = df_cells_congestion_sorted.drop_duplicates(subset="cell_name", keep="first")
    df_cells_congestion_filtered.reset_index(drop=True, inplace=True)

    # If there is congested cells then update the bands
    if df_cells_congestion_filtered.shape[0] > 0:
        df_cells_congestion_filtered["bands_upgraded"] = ""
        df_cells_congestion_filtered["tech_upgraded"] = ""

        df_cells_congestion_filtered = df_cells_congestion_filtered.merge(df_sites, how="left", on='site_id')
        unique_congested_cells = df_cells_congestion_filtered["cell_name"].unique()
        df_unique_congested_cells["cell_name"] = unique_congested_cells

        # Upgarade selection
        df_congestion = df_cells_congestion_filtered[df_cells_congestion_filtered["week_period"].astype(int) <= int(week_of_the_upgrade)]
        df_congestion = df_congestion.dropna(subset=["cell_band_available"])

        df_merged_congestion_b32 = df_congestion.merge(df_b32_aggregated, on="site_id", how="left")
        df_congestion_upgraded = df_merged_congestion_b32.apply(select_upgrade, axis=1)
        print("df_congestion_upgraded value_counts ", df_congestion_upgraded.bands_upgraded.value_counts())

    print("df_congestion_upgraded shape ", df_congestion_upgraded.shape)
    print("df_unique_congested_cells shape ", df_unique_congested_cells.shape)

    df_unique_congested_cells.to_parquet(unique_congested_cells_data_output.path)
    df_congestion_upgraded.to_parquet(congestion_data_output.path)
