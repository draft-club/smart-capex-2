import pickle
import os
import pandas as pd

from src.d00_conf.conf import conf
from src.d03_processing.OMA.new_site_modules.data_preprocessing.build_fdd_dataset import (
    prepare_density_train_dataset_fdd)
from src.d03_processing.OMA.new_site_modules.data_preprocessing.build_voice_dataset import (
    prepare_density_train_dataset_voice)
from src.d03_processing.OMA.new_site_modules.data_preprocessing.data_source import (
    add_categorical_features)
from src.d03_processing.OMA.new_site_modules.models.target_site_prediction_fdd import (
    TrafficTargetSiteModelFDD)
from src.d03_processing.OMA.new_site_modules.models.target_site_prediction_voice import (
    TrafficTargetSiteModelVoice)


def train_new_site_data():
    """
    The train_new_site_data function prepares a training dataset for a traffic prediction model,
    adds categorical features, saves the dataset, trains the model, and saves the trained model
    to a file.

    Returns
    -------
    A trained traffic prediction model saved as model_fdd_target_site.pickle.
    """
    print("rebuild train dataset fdd")
    train_dataset = prepare_density_train_dataset_fdd()
    train_dataset = add_categorical_features(train_dataset)
    train_dataset.to_pickle(os.path.join(conf["PATH"]["MODELS"],"train_dataset_fdd.pkl"))


    train_dataset = pd.read_pickle(os.path.join(conf["PATH"]["MODELS"],"train_dataset_fdd.pkl"))
    print(train_dataset.head())
    model = TrafficTargetSiteModelFDD()
    model.train(train_dataset)

    with open(os.path.join(conf["PATH"]["MODELS"],'model_fdd_target_site.pickle'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_new_site_voice():
    """
    The train_new_site_voice function prepares a training dataset for voice traffic prediction, adds
    categorical features, trains a model using this dataset, and saves the trained model to a file.

    Returns
    -------
    A trained model saved as model_voice_target_site.pickle

    """
    print("rebuild train dataset voice")
    train_dataset = prepare_density_train_dataset_voice()
    train_dataset = add_categorical_features(train_dataset)
    #train_dataset.to_pickle("data/train_dataset_voice.pkl")
    train_dataset.to_pickle(os.path.join(conf["PATH"]["MODELS"], "train_dataset_voice.pkl"))


    train_dataset = pd.read_pickle(os.path.join(conf["PATH"]["MODELS"],"train_dataset_voice.pkl"))
    model = TrafficTargetSiteModelVoice()
    model.train(train_dataset)
    with open(os.path.join(conf["PATH"]["MODELS"],
                           'model_voice_target_site.pickle'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
