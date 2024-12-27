from kfp.dsl import Dataset, Input, Model, Output, component
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def get_model(project_id:str,
              location:str,
              train_test_predictions_table_id:str,
              x_data_input: Input[Dataset],
              y_data_input: Input[Dataset],
              train_test_predictions_data_output: Output[Dataset],
              model_output:Output[Model]):
    """Train traffic improvement XGBoost regression model, predict train and test examples,
        save the predictions to a parquet file and the model artifact to Cloud staorage.

    Args:
        project_id (str): It holds the GCP project ID.
        location (str): It holds the GCP location.
        train_test_predictions_table_id (str): It holds the BigQuery table ID for train-test predictions.
        x_data_input (Input[Dataset]): It holds the input dataset containing the feature set (X).
        y_data_input (Input[Dataset]): It holds the input dataset containing the target variable (Y).
        train_test_predictions_data_output (Output[Dataset]): It holds the output dataset to store train-test predictions.
        model_output (Output[Model]): It holds the output model artifact.
    """

    import pickle
    import numpy as np
    import pandas as pd
    import pandas_gbq
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor


    df_x = pd.read_parquet(x_data_input.path)
    df_y = pd.read_parquet(y_data_input.path)

    feature_names = df_x.columns

    x_train = np.array(df_x)
    y_train = np.array(df_y["target_variable"])

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.1, random_state=50)

    model_parameters = {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 50, 'subsample': 0.5}
    model = XGBRegressor(**model_parameters)


    model.fit(x_train, y_train)
    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)


    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)

    df_train_predictions = pd.DataFrame({feature_names[0]:x_train[:,0],
                                          feature_names[1]: x_train[:,1],
                                          feature_names[2]:x_train[:,2],
                                          feature_names[3]:x_train[:,3],
                                          'y_real':y_train,
                                          'y_pred':predictions_train,
                                          'origin': "Train"})

    df_test_predictions = pd.DataFrame({feature_names[0]:x_test[:,0],
                                        feature_names[1]: x_test[:,1],
                                        feature_names[2]:x_test[:,2],
                                        feature_names[3]:x_test[:,3],
                                        'y_real':y_test,
                                        'y_pred':predictions_test,
                                        'origin': "Test"})

    df_train_test_predictions = pd.concat([df_train_predictions, df_test_predictions])

    df_train_test_predictions.to_parquet(train_test_predictions_data_output.path, index=False)
    print("df_train_test_predictions", df_train_test_predictions.info())
    pandas_gbq.to_gbq(df_train_test_predictions,
                      train_test_predictions_table_id,
                      project_id=project_id,
                      location=location,
                      if_exists='replace')
