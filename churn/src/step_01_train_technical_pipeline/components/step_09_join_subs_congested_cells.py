from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def join_subs_congested_cells(input_data_subs: Input[Dataset], input_congested_cells_m1: Input[Dataset], input_congested_cells_m2: Input[Dataset], 
                              input_congested_cells_m3: Input[Dataset], output_subs_congested_cells: Output[Dataset]
            ):
    
    import pandas as pd 
    
    data_subs = pd.read_parquet(input_data_subs.path)
    nb_cell_cong_m1 = pd.read_parquet(input_congested_cells_m1.path)
    nb_cell_cong_m2 = pd.read_parquet(input_congested_cells_m2.path)
    nb_cell_cong_m3 = pd.read_parquet(input_congested_cells_m3.path)
    
    data_subs = data_subs.join(nb_cell_cong_m1, on="MAIN_SITE_CODE_NEW", rsuffix = "_m1")
    data_subs = data_subs.join(nb_cell_cong_m2, on="MAIN_SITE_CODE_NEW", rsuffix = "_m2")
    data_subs = data_subs.join(nb_cell_cong_m3, on="MAIN_SITE_CODE_NEW", rsuffix = "_m3")
    
    data_subs = data_subs.rename({"nb_cell_cong": "nb_cell_cong_m1"}, axis =1)
    data_subs[["nb_cell_cong_m1", "nb_cell_cong_m2", "nb_cell_cong_m3"]] = data_subs[["nb_cell_cong_m1", "nb_cell_cong_m2", "nb_cell_cong_m3"]].fillna(0)
    
    data_subs["nb_cell_cong_diff_m2_m3"] = data_subs["nb_cell_cong_m2"] - data_subs["nb_cell_cong_m3"]
    data_subs["nb_cell_cong_diff_m1_m2"] = data_subs["nb_cell_cong_m1"] - data_subs["nb_cell_cong_m2"]
    
    data_subs.to_parquet(output_subs_congested_cells.path)