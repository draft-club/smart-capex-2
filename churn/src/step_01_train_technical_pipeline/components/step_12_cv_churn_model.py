from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config


@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def cv_churn_model(input_dataset_train: Input[Dataset], input_dataset_test: Input[Dataset]):
    
    import pandas as pd 
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate
    
    TARGET = "SUBS_STATUS_DESC"
    
    dataset_train = pd.read_parquet(input_dataset_train.path)
    dataset_test = pd.read_parquet(input_dataset_test.path)
    
    features = [f for f in dataset_train.columns if f != TARGET]
    
    base_model = LogisticRegression(max_iter=1000)
    sc_features = StandardScaler()

    model = Pipeline([("scaler", sc_features), 
                                    ('estimator', base_model)])
    cv = cross_validate(model, dataset_train[features] , dataset_train[TARGET], scoring = ["accuracy", "roc_auc"], return_estimator =True)

    print("acc : ", np.mean(cv["test_accuracy"]))
    print("auc roc : ", np.mean(cv["test_roc_auc"])) # debug perf
    print(cv["test_accuracy"])
    print(cv["test_roc_auc"])
    