from kfp.dsl import Dataset, Input, Model, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def evaluate_model(train_test_predictions_data_input: Input[Dataset],
                   model_input: Input[Model]):
    """Evaluate traffic improvement XGBoost model and print the resulted evaluation metrics.

    Args:
        train_test_predictions_data_input (Input[Dataset]): It holds the input dataset containing
                                                            train and test predictions dataframe path.
        model_input (Input[Model]): It holds the input model artifact including model path.
    """

    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score

    df_train_test_predictions = pd.read_parquet(train_test_predictions_data_input.path)

    # Load the model
    with open(model_input.path, 'rb') as f:
        model = pickle.load(f)

    df_train_predictions = df_train_test_predictions[df_train_test_predictions['origin'] == 'Train']
    df_test_predictions = df_train_test_predictions[df_train_test_predictions['origin'] == 'Test']

    # MAE
    errors_train = abs(df_train_predictions['y_pred'] - df_train_predictions['y_real'])
    errors_test = abs(df_test_predictions['y_pred'] - df_test_predictions['y_real'])
    mean_error_train = round(np.mean(errors_train), 2)
    mean_error_test = round(np.mean(errors_test), 2)

    # MAPE
    mape_test = 100 * np.mean(errors_test / df_test_predictions['y_real'])
    r2_score_train = r2_score(df_train_predictions['y_real'], df_train_predictions['y_pred'])
    r2_score_test = r2_score(df_test_predictions['y_real'], df_test_predictions['y_pred'])

    dict_metrics = {"mean_error_train": mean_error_train,
                    "mean_error_test": mean_error_test,
                    "mape_test": mape_test,
                    "r2_score_train": r2_score_train,
                    "r2_score_test": r2_score_test,
                    "feature_importance": model.feature_importances_}

    print(dict_metrics)
