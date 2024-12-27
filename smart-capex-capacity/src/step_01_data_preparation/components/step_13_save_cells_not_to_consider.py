from kfp.dsl import Dataset, Input, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def save_cells_not_to_consider(project_id: str,
                               location: str,
                               cells_not_to_consider_table_id: str,
                               cells_not_to_consider_recent_cells: Input[Dataset],
                               cells_not_to_consider_unmounted_cells: Input[Dataset],
                               cells_not_to_consider_with_missing_weeks: Input[Dataset],
                               cells_not_to_consider_with_high_variation_in_kpi: Input[Dataset]):
    """Save cells not to consider dataframe to BigQuery.

    Args:
        project_id (str): It holds the project ID of GCP.
        location (str): It holds the location assigned to the project on GCP.
        cells_not_to_consider_table_id (str): It holds the table ID for the cells not to consider in BigQuery.
        cells_not_to_consider_recent_cells (Input[Dataset]): It holds the input dataset containing the cells not to consider
                                                                for being recent.
        cells_not_to_consider_unmounted_cells (Input[Dataset]): It holds the input dataset containing cells not to consider
                                                                    for being unmounted.
        cells_not_to_consider_with_missing_weeks (Input[Dataset]): It holds the input dataset containing cells not to
                                                                    consider for including missing weeks.
        cells_not_to_consider_with_high_variation_in_kpi (Input[Dataset]): It holds the input dataset containing cells not 
                                                                            to consider due to high variation in KPI.
    """

    # Imports
    import pandas as pd
    import pandas_gbq

    # Load Data
    df_unique_recent_cells = pd.read_parquet(cells_not_to_consider_recent_cells.path)
    print("df_recent_cells:", len(df_unique_recent_cells))

    df_unique_unmounted_cells = pd.read_parquet(cells_not_to_consider_unmounted_cells.path)
    print("df_unmounted_cells:", len(df_unique_unmounted_cells))

    df_unique_cells_with_missing_weeks = pd.read_parquet(cells_not_to_consider_with_missing_weeks.path)
    print("df_cells_with_missing_weeks:", len(df_unique_cells_with_missing_weeks))

    df_unique_cells_with_high_variation_kpi = pd.read_parquet(cells_not_to_consider_with_high_variation_in_kpi.path)
    print("df_cells_with_high_variation_kpi:", len(df_unique_cells_with_high_variation_kpi))

    df_max_shape = max([df_unique_recent_cells,
                    df_unique_unmounted_cells,
                    df_unique_cells_with_missing_weeks,
                    df_unique_cells_with_high_variation_kpi], key=lambda df: df.shape[0])

    max_range = range(df_max_shape.shape[0])

    df_unique_recent_cells = df_unique_recent_cells.reindex(max_range)
    df_unique_unmounted_cells = df_unique_unmounted_cells.reindex(max_range)
    df_unique_cells_with_missing_weeks = df_unique_cells_with_missing_weeks.reindex(max_range)
    df_unique_cells_with_high_variation_kpi = df_unique_cells_with_high_variation_kpi.reindex(max_range)

    dict_cells_not_to_consider = {
    "recent_cells": df_unique_recent_cells["cell_name"].tolist(),
    "unmounted_cells": df_unique_unmounted_cells["cell_name"].tolist(),
    "cells_with_missing_periods": df_unique_cells_with_missing_weeks["cell_name"].tolist(),
    "cells_with_high_variation_kpi": df_unique_cells_with_high_variation_kpi["cell_name"].tolist()}

    df_cells_not_to_consider = pd.DataFrame.from_dict(dict_cells_not_to_consider)
    print("cells_not_to_consider shape", df_cells_not_to_consider.shape)

    # Save to bigquery
    pandas_gbq.to_gbq(df_cells_not_to_consider,
                      cells_not_to_consider_table_id,
                      project_id=project_id,
                      location=location, if_exists='replace')
