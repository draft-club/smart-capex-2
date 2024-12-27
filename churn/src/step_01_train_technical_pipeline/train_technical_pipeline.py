from kfp import dsl
from utils.config import pipeline_config
from components.step_01_get_subs_data import get_data_subs
from components.step_02_get_kpis_data import get_data_kpis
from components.step_03_join_kpis import join_kpis
from components.step_04_join_subs_kpis import join_subs_kpis
from components.step_05_get_capacity_kpis_data import get_capacity_kpis_data
from components.step_06_join_capacity_kpis import join_capacity_kpis
from components.step_07_join_subs_capacity_kpis import join_subs_capacity_kpis
from components.step_08_get_congested_cells import get_congested_cells
from components.step_09_join_subs_congested_cells import join_subs_congested_cells
from components.step_10_create_dataset import create_dataset
from components.step_11_select_features import select_features
from components.step_12_cv_churn_model import cv_churn_model

@dsl.pipeline(name='churn-prediction-pipeline',
    pipeline_root=pipeline_config["pipeline_root"],
    description="churn prediction pipeline")
def pipeline(project_id: str, location: str, m0: int, dataset_train_table_id: str, 
             nb_train_samples_per_class: int,
             nb_test_samples_per_class: int):
    
    get_data_subs_op = get_data_subs(project_id=project_id, location=location, m0 = m0).set_caching_options(True)
    get_data_m1_kpis_op = get_data_kpis(project_id=project_id, location=location, m0 = m0, delta = 1).set_caching_options(True)
    get_data_m3_kpis_op = get_data_kpis(project_id=project_id, location=location, m0 = m0, delta = 3).set_caching_options(True)

    join_kpis_op = join_kpis(input_m1_kpis = get_data_m1_kpis_op.outputs["kpis_output"]
                                       , input_m3_kpis = get_data_m3_kpis_op.outputs["kpis_output"]).set_caching_options(True)
                                                                                                          
    join_subs_kpis_op = join_subs_kpis(input_data_subs = get_data_subs_op.outputs["data_subs_output"], 
                                       input_kpis= join_kpis_op.outputs["output_kpis"]).set_caching_options(True)
    
    get_m1_congestion_kpis_op = get_capacity_kpis_data(project_id=project_id, location=location, m0 = m0, delta= 1).set_caching_options(True)
    get_m3_congestion_kpis_op = get_capacity_kpis_data(project_id=project_id, location=location, m0 = m0, delta = 3).set_caching_options(True)
    
    join_congestion_kpis_op = join_capacity_kpis(input_m1_congestion_kpis = get_m1_congestion_kpis_op.outputs["congestion_kpis_output"]
                                       , input_m3_congestion_kpis = get_m3_congestion_kpis_op.outputs["congestion_kpis_output"]).set_caching_options(True)
    
    join_subs_congestion_kpis_op = join_subs_capacity_kpis(input_data_subs = join_subs_kpis_op.outputs["subs_kpis"], 
                                       input_congestion_kpis= join_congestion_kpis_op.outputs["output_congestion_kpis"]).set_caching_options(True)
    
    get_m1_congested_cells_op = get_congested_cells(project_id=project_id, location=location, m0 = m0, delta = 1).set_caching_options(True)
    get_m2_congested_cells_op = get_congested_cells(project_id=project_id, location=location, m0 = m0, delta = 2).set_caching_options(True)
    get_m3_congested_cells_op = get_congested_cells(project_id=project_id, location=location, m0 = m0, delta = 3).set_caching_options(True)
    
    join_subs_congested_cells_op = join_subs_congested_cells(input_data_subs=join_subs_congestion_kpis_op.outputs["output_subs_congestion_kpis"], 
                                                             input_congested_cells_m1= get_m1_congested_cells_op.outputs['congested_cells'],
                                                             input_congested_cells_m2= get_m2_congested_cells_op.outputs['congested_cells'],
                                                             input_congested_cells_m3= get_m3_congested_cells_op.outputs['congested_cells']).set_caching_options(True)
    
    create_dataset_op = create_dataset(input_data_subs=join_subs_congested_cells_op.outputs["output_subs_congested_cells"]).set_caching_options(True)
    
    select_features_op = select_features(input_dataset_train=create_dataset_op.outputs['output_dataset_train']).set_caching_options(True)
    cv_churn_model(input_dataset_train = select_features_op.outputs["dataset_train_selected_features"],
                  input_dataset_test = create_dataset_op.outputs["output_dataset_test"])  
    
