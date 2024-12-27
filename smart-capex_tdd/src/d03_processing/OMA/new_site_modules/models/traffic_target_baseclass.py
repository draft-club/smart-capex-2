from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataPreprocessor:  # pylint: disable=too-few-public-methods
    """
    The DataPreprocessor class initializes two StandardScaler objects for scaling features and
    target variables in a dataset.
    """
    def __init__(self):
        self.sc_features = StandardScaler()
        self.sc_target = StandardScaler()


class TrafficTargetSiteModelBase:
    """
    The TrafficTargetSiteModelBase class is designed to model and predict traffic data using machine
     learning techniques. It initializes a data preprocessor, an XGBoost regressor, and a pipeline
     for transforming and fitting the data. The class includes methods for training the model using
     grid search, training with a specific dataset, and making predictions

    """
    def __init__(self, model_type='linear'):
        self.model = None
        self.target = "target_traffic"
        self.preprocessor = DataPreprocessor()
        self.xgb = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8,
                                learning_rate=0.1,
                                max_depth=3, n_estimators=50, subsample=0.8)
        self.ttr_linear = TransformedTargetRegressor(regressor=self.xgb,
                                                     transformer=self.preprocessor.sc_target)
        self.pipeline = Pipeline(
            steps=[('transformer', self.preprocessor.sc_features), ("estimator", self.ttr_linear)])
        self.features = []
        print(model_type)

    def _compute_estim(self, dataset):
        """
        The _compute_estim method is an abstract method in the TrafficTargetSiteModelBase class.
        It is intended to be overridden by subclasses to provide specific implementations for
        computing estimations on the dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            A pandas DataFrame containing the data to be processed.

        Returns
        -------
        Raises NotImplementedError if not overridden by a subclass
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def _forward(self, x):
        """
        The _forward method in the TrafficTargetSiteModelBase class processes the input data by
        first computing estimations using the _compute_estim method. It then extracts the relevant
        features and uses the model to predict the target variable.

        Parameters
        ----------
        x: pd.DataFrame
            A pandas DataFrame containing the data to be processed

        Returns
        -------
        Returns the predicted values as a numpy array
        """
        x = self._compute_estim(x)
        x_feature = x[self.features]
        return self.model.predict(x_feature)

    def train_grid_search(self, dataset, param_grid):
        """
        The train_grid_search method in the TrafficTargetSiteModelBase class performs hyperparameter
        tuning using grid search with cross-validation. It preprocesses the dataset, splits it into
        stratified bins, and then uses GridSearchCV to find the best hyperparameters for the model
        pipeline.

        Parameters
        ----------
        dataset: pd.DataFrame
            A pandas DataFrame containing the traffic data to be processed.
        param_grid: dict
            A dictionary specifying the hyperparameters to be tuned and their respective ranges.
        Returns
        -------
        Prints the best hyperparameters and the best score from the grid search
        """
        dataset = self._compute_estim(dataset)
        dataset = dataset.dropna()
        dataset = dataset[dataset.target_traffic <= 6000]
        _, bins = np.histogram(dataset.target_traffic, bins='doane')
        dataset["bin"] = pd.cut(dataset["target_traffic"], bins, labels=False)
        s = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        dataset = dataset.dropna()
        s.get_n_splits(dataset, dataset.bin)
        cv = GridSearchCV(self.pipeline, param_grid, scoring="neg_mean_squared_error",
                          cv=s.split(dataset, dataset.bin), verbose=4)
        cv.fit(dataset[self.features], dataset[self.target])
        print(cv.best_params_)
        print(cv.best_score_)

    def train(self, dataset):
        """
        The train method in the TrafficTargetSiteModelBase class preprocesses the dataset, splits it
        into training and testing sets, trains the model on the training set, and evaluates it on
        the testing set. It prints the R-squared and Mean Absolute Percentage Error (MAPE) scores
        for the test set.

        Parameters
        ----------
        dataset: pd.DataFrame
            A pandas DataFrame containing the traffic data to be processed.
        Returns
        -------
        Prints the R-squared and MAPE scores for the test set.
        """
        dataset = self._compute_estim(dataset)
        dataset = dataset.dropna()
        dataset = dataset[dataset.target_traffic <= 6000]
        _, bins = np.histogram(dataset.target_traffic, bins='doane')
        dataset["bin"] = pd.cut(dataset["target_traffic"], bins, labels=False)
        s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(s.split(dataset, dataset.bin))
        dataset_train = dataset.iloc[train_idx].assign(residuals=[0] * len(train_idx))
        dataset_train = self._train_step(dataset_train, 20, -20, print_graph=True)
        self.model = self.pipeline
        dataset_test = dataset.iloc[test_idx]
        y_pred = self.model.predict(dataset_test[self.features])
        r2 = r2_score(dataset_test[self.target], y_pred)
        mape = mean_absolute_percentage_error(dataset_test[self.target], y_pred)
        print("test scores, r2 ", r2, " mape :", mape)

    def _train_step(self, dataset, residual_max, residual_min, print_graph=False):
        """
        The _train_step method in the TrafficTargetSiteModelBase class filters the dataset based on
        residuals, fits the model pipeline, and evaluates its performance. It prints the R-squared
        and Mean Absolute Percentage Error (MAPE) scores, calculates residuals, and optionally
        plots them

        Parameters
        ----------
        dataset: pd.DataFrame
            A pandas DataFrame containing the data to be processed.
        residual_max: int
            Maximum allowable residual value for filtering the dataset
        residual_min: int
            Minimum allowable residual value for filtering the dataset
        print_graph: bool
            A boolean flag to indicate whether to plot residuals.

        Returns
        -------
        dataset: pd.DataFrame
            Returns the updated dataset with residuals and predicted values
        """
        dataset = dataset[(dataset.residuals <= residual_max) & (dataset.residuals >= residual_min)]
        x_feature = dataset[self.features]
        y = dataset[self.target]
        self.pipeline.fit(x_feature, y)
        y_pred = self.pipeline.predict(x_feature)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        print("train scores, r2 ", r2, " mape :", mape)
        residuals = y - y_pred
        print(np.mean(residuals))
        if print_graph:
            plt.scatter(list(range(len(residuals))), residuals)
            plt.show()
        dataset["residuals"] = residuals
        dataset['y_pred'] = y_pred
        return dataset

    def test(self):
        pass

    def predict(self, x):
        """
        The predict method in the TrafficTargetSiteModelBase class processes input data to generate
        traffic predictions. It uses the _forward method to compute the predictions and returns
        them in a pandas DataFrame.

        Parameters
        ----------
        x: pd.DataFrame
            A pandas DataFrame containing the input data to be processed, including a site column
            and feature columns.

        Returns
        -------
        A pandas DataFrame with columns site_id and predicted_traffic, containing the site
        identifiers and their corresponding predicted traffic values.

        """
        y = self._forward(x)
        return pd.DataFrame({"site_id": x.site, "predicted_traffic": y.tolist()})
