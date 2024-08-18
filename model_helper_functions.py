"""Imports Tuple type"""

from typing import Tuple

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from data_helper_functions import *
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from typing import Optional, Dict, Any
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def create_pipeline(
    df: pd.DataFrame,
    model,
) -> Tuple[
    Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], np.ndarray
]:
    """
    Creates a machine learning pipeline with preprocessing steps and a given model.

    Parameters
    ----------
    df : pandas.DataFrame
        All model data
    model : estimator object
        The machine learning model to be used in the pipeline.

    Returns
    -------
    Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list of str]
        A tuple containing the pipeline, training features, training target, test features,
        test target, and the list of preprocessed feature names.

    Data Fields:
    ------------
    AVG_RECALL_IMMEDIATE_TOTAL : float
        Average immediate recall total score. [0..25]
    AVG_RECALL_DELAYED_TOTAL : float
        Average delayed recall total score. [0..25]
    AVG_RAV_LISTA_1_RECALL : float
        Average RAVLT list A trial 1 recall. [0..15]
    AVG_RAV_LISTA_1_ERRORS : float
        Average RAVLT intrusions for List A Trial 1 [0..?]
    AVG_RAV_LISTA_5_RECALL : float
        Average RAVLT list A trial 5 recall. [0..15]
    AVG_RAV_LISTA_5_ERRORS : float
        Average RAVLT intrusions for list A Trial 5 [0..?]
    AVG_RAV_LISTB_RECALL : float
        Average RAVLT list B recall. [0..15]
    AVG_RAV_LISTB_ERRORS : float
        Average RAVLT intrusions. [0..?]
    AVG_CLOCK_SCORE : int
        Average clock drawing test score [0-5].
    AVG_CLOCK_COPY : int
        Average clock copying test score [0-5].
    GENDER : int
        Gender (categorical).
    MARRIAGE_STATUS : int
        Marriage status (categorical).
    EDUCATION : int
        Education level [0..20].
    HOME_TYPE : int
        Home type (categorical).
    ETHNICITY : int
        Ethnicity (categorical).
    AGE : float
        Age of the participant.
    DIAGNOSIS : int
        Diagnosis of the participant (target variable).
    """

    # Features and target
    features = [
        "AVG_RECALL_IMMEDIATE_TOTAL",
        "AVG_RECALL_DELAYED_TOTAL",
        "AVG_RAV_LISTA_1_RECALL",
        "AVG_RAV_LISTA_1_ERRORS",
        "AVG_RAV_LISTA_5_RECALL",
        "AVG_RAV_LISTA_5_ERRORS",
        "AVG_RAV_LISTB_RECALL",
        "AVG_RAV_LISTB_ERRORS",
        "AVG_CLOCK_SCORE",
        "AVG_CLOCK_COPY",
        "GENDER",
        "MARRIAGE_STATUS",
        "EDUCATION",
        "HOME_TYPE",
        "ETHNICITY",
        "AGE",
    ]
    target = "DIAGNOSIS"

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    # Count the number of NaN values in each row before the bfill operation
    nan_counts_before_train = x_train.isna().sum(axis=1)
    nan_counts_before_test = x_test.isna().sum(axis=1)

    # Forward fill missing values in training and test sets separately
    x_train = x_train.ffill()
    x_test = x_test.ffill()

    # Perform the bfill operation
    x_train = x_train.bfill()
    x_test = x_test.bfill()

    # Count the number of NaN values in each row after the bfill operation
    nan_counts_after_train = x_train.isna().sum(axis=1)
    nan_counts_after_test = x_test.isna().sum(axis=1)

    # Calculate the number of rows that had values filled
    rows_filled_train = (nan_counts_before_train > nan_counts_after_train).sum()
    rows_filled_test = (nan_counts_before_test > nan_counts_after_test).sum()

    # Calculate the percentage of total rows that had values filled
    percentage_filled_train = (rows_filled_train / len(x_train)) * 100
    percentage_filled_test = (rows_filled_test / len(x_test)) * 100

    print(
        f"{model}: % of missing training set values filled: {rows_filled_train} ({percentage_filled_train.round(2)}%)"
    )
    print(
        f"{model}: % of missing test set values filled: {rows_filled_test} ({percentage_filled_test.round(2)}%)"
    )

    # Define categorical and numerical features
    categorical_features = [
        "GENDER",
        "MARRIAGE_STATUS",
        "HOME_TYPE",
        "ETHNICITY",
        "EDUCATION",
    ]
    ordinal_features = ["AVG_CLOCK_SCORE", "AVG_CLOCK_COPY"]
    numerical_features = [
        "AVG_RECALL_IMMEDIATE_TOTAL",
        "AVG_RECALL_DELAYED_TOTAL",
        "AVG_RAV_LISTA_1_RECALL",
        "AVG_RAV_LISTA_5_RECALL",
        "AVG_RAV_LISTB_RECALL",
    ]
    numerical_inverted_features = [
        "AVG_RAV_LISTA_1_ERRORS",
        "AVG_RAV_LISTA_5_ERRORS",
        "AVG_RAV_LISTB_ERRORS",
        "AGE",
    ]

    # Define known categories for categorical and ordinal features
    known_categories = {
        "GENDER": sorted(df["GENDER"].unique()),
        "MARRIAGE_STATUS": sorted(df["MARRIAGE_STATUS"].unique()),
        "HOME_TYPE": sorted(df["HOME_TYPE"].unique()),
        "ETHNICITY": sorted(df["ETHNICITY"].unique()),
        "EDUCATION": sorted(df["EDUCATION"].unique()),
        "AVG_CLOCK_SCORE": sorted(df["AVG_CLOCK_SCORE"].unique()),
        "AVG_CLOCK_COPY": sorted(df["AVG_CLOCK_COPY"].unique()),
    }

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[
                        known_categories["GENDER"],
                        known_categories["MARRIAGE_STATUS"],
                        known_categories["HOME_TYPE"],
                        known_categories["ETHNICITY"],
                        known_categories["EDUCATION"],
                    ],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
            (
                "scaler",
                MinMaxScaler(),
            ),
        ]
    )

    # Preprocessing pipeline for ordinal features
    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        known_categories["AVG_CLOCK_SCORE"],
                        known_categories["AVG_CLOCK_COPY"],
                    ]
                ),
            ),
            (
                "scaler",
                MinMaxScaler(),
            ),
        ]
    )

    # Preprocessing pipeline for numerical features
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "scaler",
                MinMaxScaler(),
            ),
        ]
    )

    # Invert columns so that factors have the same directionality
    invert_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),  # Ensure no NaNs before inverting
            (
                "inverter",
                FunctionTransformer(
                    lambda X: invert_columns(
                        pd.DataFrame(X, columns=numerical_inverted_features),
                        numerical_inverted_features,
                    )
                ),
            ),
            (
                "scaler",
                MinMaxScaler(),
            ),
        ]
    )

    # Combine preprocessing steps, including inversion for specified columns
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_transformer, categorical_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("num", numerical_transformer, numerical_features),
            ("inv", invert_transformer, numerical_inverted_features),
        ]
    )

    # Create and fit the preprocessing pipeline
    preprocessor.fit(x_train)

    # Transform the training and test data

    x_train_preprocessed = preprocessor.transform(x_train)

    x_test_preprocessed = preprocessor.transform(x_test)

    # Create a DataFrame for the preprocessed data (for inspection)
    preprocessed_feature_names = (
        [x for x in preprocessor.named_transformers_["cat"].get_feature_names_out()]
        + ordinal_features
        + numerical_features
        + numerical_inverted_features
    )

    # # Create DataFrames for the preprocessed data
    x_train_preprocessed_df = pd.DataFrame(x_train_preprocessed, columns=preprocessed_feature_names)  # type: ignore
    x_test_preprocessed_df = pd.DataFrame(x_test_preprocessed, columns=preprocessed_feature_names)  # type: ignore

    # # Print the preprocessed training data
    # print("Preprocessed Training Data:")
    # print(x_train_preprocessed_df.head)

    # # Print the preprocessed test data
    # print("Preprocessed Test Data:")
    # print(x_test_preprocessed_df.head)

    # Create the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    return pipeline, x_train, y_train, x_test, y_test, preprocessed_feature_names, x_train_preprocessed  # type: ignore


def invert_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Inverts the values of specified columns in a DataFrame.
    The inversion is performed by subtracting each value from the maximum value in its respective column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be inverted.
    cols : list of str
        The list of column names to invert.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified columns inverted.
    """
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            max_val = df_copy[col].max()
            df_copy[col] = max_val - df_copy[col]
    return df_copy


class ModelMetrics:
    """
    A class to encapsulate metrics for evaluating model success across multiple runs.

    Attributes
    ----------
    metrics : dict
        A dictionary to store various metrics for three runs of a model.

    Methods
    -------
    update_classification_metrics(run: int, y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None) -> None
        Updates classification metrics for a given run.
    update_regression_metrics(run: int, y_true: pd.Series, y_pred: pd.Series) -> None
        Updates regression metrics for a given run.
    to_dataframe() -> pd.DataFrame
        Converts the stored metrics (excluding classification report and confusion matrix) to a DataFrame.
    print_reports() -> None
        Prints classification reports and confusion matrices for each run.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the ModelMetrics class with a dictionary to store metrics for three runs.

        Parameters
        ----------
        name : str
            The name of the model.

        """
        self.name = name
        self.metrics: Dict[str, Dict[int, Any]] = {
            "accuracy": {1: None, 2: None, 3: None},
            "precision": {1: None, 2: None, 3: None},
            "recall": {1: None, 2: None, 3: None},
            "f1_score": {1: None, 2: None, 3: None},
            "roc_auc": {1: None, 2: None, 3: None},
            "log_loss": {1: None, 2: None, 3: None},
        }

    def update_metrics(
        self,
        run: int,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: Optional[pd.Series],
    ) -> None:
        """
        Updates metrics for a given run based on the specified task.

        Parameters
        ----------
        run : int
            The run number (1, 2, or 3).
        y_true : pd.Series
            True labels or values.
        y_pred : pd.Series
            Predicted labels or values.
        y_prob : Optional[pd.Series], optional
            Predicted probabilities, by default None.

        Returns
        -------
        None
        """
        self.metrics["accuracy"][run] = accuracy_score(y_true, y_pred)
        self.metrics["precision"][run] = precision_score(
            y_true, y_pred, average="weighted"
        )
        self.metrics["recall"][run] = recall_score(y_true, y_pred, average="weighted")
        self.metrics["f1_score"][run] = f1_score(y_true, y_pred, average="weighted")
        self.metrics["roc_auc"][run] = roc_auc_score(y_true, y_prob, multi_class="ovo", average="weighted")  # type: ignore
        self.metrics["log_loss"][run] = log_loss(y_true, y_prob)  # type: ignore

    def metric_to_dataframe(self, metric_name: str) -> pd.DataFrame:
        """
        Converts the specified metric to a DataFrame.

        Parameters
        ----------
        metric_name : str
            The name of the metric to convert to a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified metric for each run.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found in metrics.")

        metric_data = self.metrics[metric_name]
        if not metric_data:
            raise ValueError(f"No data available for metric {metric_name}.")

        df = pd.DataFrame.from_dict(metric_data, orient="index", columns=[metric_name])
        return df


