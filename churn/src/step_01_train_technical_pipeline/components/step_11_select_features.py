from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def select_features(input_dataset_train: Input[Dataset], dataset_train_selected_features: Output[Dataset]):
    
    import pandas as pd 
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SequentialFeatureSelector

    dataset_train = pd.read_parquet(input_dataset_train.path)
    
    features = ["SMS_SENT_CNT", "SMS_RECEIVED_CNT", "CALLS_RECEIVED_SEC", "CALLS_MADE_SEC", "DATA_TRAFFIC_KB", "INT_CALL_COST_MIN", "CALLS_MADE_OFFNET_CNT", "CALLS_RECEIVED_OFFNET_CNT", "FREE_SERVICES_CNT", "OVERBUNDLE_AMT_EUR", 'SMS_SENT_CNT_diff', 'SMS_RECEIVED_CNT_diff', 'CALLS_RECEIVED_SEC_diff', 'CALLS_MADE_SEC_diff', 'DATA_TRAFFIC_KB_diff', 'INT_CALL_COST_MIN_diff', 'CALLS_MADE_OFFNET_CNT_diff', 'CALLS_RECEIVED_OFFNET_CNT_diff', 'FREE_SERVICES_CNT_diff', 'OVERBUNDLE_AMT_EUR_diff',
           'average_nb_user_in_queue', 'average_throughput_user_dl','average_nb_user_in_queue_m3', 'average_throughput_user_dl_m3','average_nb_user_in_queue_diff', 'average_throughput_user_dl_diff', "nb_cell_cong_m1", "nb_cell_cong_m2", "nb_cell_cong_m3", "nb_cell_cong_diff_m1_m2", "nb_cell_cong_diff_m2_m3", 'max_nb_user_in_queue_diff', "max_nb_user_in_queue", "max_nb_user_in_queue_m3", 'min_throughput_user_dl_diff', "min_throughput_user_dl", "min_throughput_user_dl_m3"]
    
    TARGET = "SUBS_STATUS_DESC"
    
    print(dataset_train[TARGET].unique())
    base_model = LogisticRegression(max_iter=1000)
    sc_features = StandardScaler()

    fs = SequentialFeatureSelector(base_model, n_features_to_select = "auto", tol = 10e-5)
    model = Pipeline([("scaler", sc_features), 
                                    ('feature_selection', fs)])

    model.fit( dataset_train[features], dataset_train[TARGET])
    selected_features = []
    for f, b in zip(features, model["feature_selection"].support_):
        if b:
            print(f)
            selected_features.append(f)
            
    dataset_train[selected_features + [TARGET]].to_parquet(dataset_train_selected_features.path)
    #dataset_train[features + [target]].to_parquet(dataset_train_selected_features.path)