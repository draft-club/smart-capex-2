import os
import pickle

from src.d00_conf.conf import conf
from src.d03_processing.OMA.new_site_modules.data_preprocessing.build_fdd_dataset import (
    prepare_density_pred_dataset_fdd)
from src.d03_processing.OMA.new_site_modules.data_preprocessing.build_voice_dataset import (
    prepare_density_pred_dataset_voice)
from src.d03_processing.OMA.new_site_modules.data_preprocessing.data_source import (
    add_categorical_features)


def apply_new_site_data():
    """
    The apply_new_site_data function prepares a dataset for predicting traffic density at new sites,
    adds categorical features, loads a pre-trained model, and generates traffic predictions.
    The results are saved to CSV files.

    Returns
    -------
    traffic_predictions: pd.DataFrame
        Traffic predictions as a DataFrame
    """
    path = os.path.join(conf['PATH']['RANDIM'], "densification_result_FDD_from_capacity.xlsx")
    pred_dataset = prepare_density_pred_dataset_fdd(path, fdd_config=[3, 3, 3, 3, 3])
    pred_dataset = add_categorical_features(pred_dataset, True, path)
    with open(os.path.join(conf["PATH"]["MODELS"],'model_fdd_target_site.pickle'), 'rb') as handle:
        model = pickle.load(handle)

    pred_dataset.to_csv("data/density_prediction_dataset_fdd.csv", header=True, index=False)
    pred_dataset.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                     "density_prediction_dataset_fdd.csv"),
                        header=True, index=False)
    traffic_predictions = model.predict(pred_dataset)
    print(traffic_predictions.head())
    traffic_predictions.to_csv("data/traffic_prediction_density_fdd.csv", header=True, index=False)
    traffic_predictions.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                            "traffic_prediction_density_fdd.csv"),
                               header=True, index=False)
    return traffic_predictions



def apply_new_site_voice():
    """
    The apply_new_site_voice function prepares a dataset for predicting voice traffic density at new
    sites, applies a pre-trained model to make predictions, and saves the results to CSV files.

    Returns
    -------
    traffic_predictions: pandas.DataFrame
        A DataFrame containing the predicted traffic density values.
    """
    print("Pred voice")
    path = "data/densification_result_FDD_from_capacity.xlsx"
    path = os.path.join(conf['PATH']['RANDIM'], "densification_result_FDD_from_capacity.xlsx")
    pred_dataset = prepare_density_pred_dataset_voice(path, fdd_config=[3, 3, 3, 3, 3])
    pred_dataset = add_categorical_features(pred_dataset, True, path)
    with open(os.path.join(conf["PATH"]["MODELS"],
                           'model_voice_target_site.pickle'), 'rb') as handle:
        model = pickle.load(handle)

    pred_dataset.to_csv("data/density_prediction_dataset_voice.csv", header=True, index=False)
    pred_dataset.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                     "density_prediction_dataset_voice.csv"),
                        header=True, index=False)
    traffic_predictions = model.predict(pred_dataset)
    print(traffic_predictions.head())
    traffic_predictions.to_csv("data/traffic_prediction_density_voice.csv", header=True,
                               index=False)
    traffic_predictions.to_csv(os.path.join(conf['PATH']['MODELS_OUTPUT'],
                                            "traffic_prediction_density_voice.csv"),
                               header=True, index=False)
    return traffic_predictions
