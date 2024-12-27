from kfp.dsl import Dataset, Input, Output, component
from utils.config import pipeline_config

@component(base_image="europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.3.0")
def create_dataset(input_data_subs: Input[Dataset], nb_train_samples_per_class: int,
                   nb_test_samples_per_class: int,
                   output_dataset_train: Output[Dataset], output_dataset_test: Output[Dataset]
            ):
    
    import pandas as pd
    
    data_subs = pd.read_parquet(input_data_subs.path)

    data_subs_off = data_subs[data_subs.SUBS_STATUS_DESC == "Switched Off"]
    print(len(data_subs_off))
    data_subs_off.sample(frac= 1) 
    data_subs_off_train = data_subs_off[:nb_train_samples_per_class]
    data_subs_off_test = data_subs_off[nb_train_samples_per_class:nb_train_samples_per_class +nb_test_samples_per_class]

    data_subs_active = data_subs[data_subs.SUBS_STATUS_DESC == "Active Subscriber"]
    print(len(data_subs_active))
    data_subs_active.sample(frac= 1) 
    data_subs_active_train = data_subs_active[:nb_train_samples_per_class]
    data_subs_active_test = data_subs_active[nb_train_samples_per_class:nb_train_samples_per_class +nb_test_samples_per_class]

    dataset_train = pd.concat([data_subs_off_train, data_subs_active_train]) 
    dataset_train.SUBS_STATUS_DESC = dataset_train.SUBS_STATUS_DESC.apply(lambda x: 1 if x=="Active Subscriber" else 0)
    dataset_train.to_parquet(output_dataset_train.path)

    dataset_test = pd.concat([data_subs_off_test, data_subs_active_test]) 
    dataset_test.SUBS_STATUS_DESC = dataset_test.SUBS_STATUS_DESC.apply(lambda x: 1 if x=="Active Subscriber" else 0)
    dataset_test.to_parquet(output_dataset_test.path)