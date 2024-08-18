import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
)
from model_helper_functions import *
from plotting_helper_functions import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
)


def run_logistic_regression_model(
    df: pd.DataFrame, year: int, metrics: ModelMetrics
) -> None:
    """
    Trains and evaluates a Logistic Regression model using the given data and year for reference.

    This function creates a machine learning pipeline, performs hyperparameter tuning using GridSearchCV,
    evaluates the model on the test set, and generates several plots to visualize the results.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data for model training and testing.
    year : int
        The year used as a reference for labeling plots and reports.
    metrics : ModelMetrics
        Class to store all metrics related to the model.

    Returns
    -------
    None
    """
    (
        pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessed_feature_names,
        x_train_preprocessed,
    ) = create_pipeline(df, LogisticRegression())

    # Define the parameters for GridSearch
    param_grid = [
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100, 1000],
            "classifier__penalty": ["l1"],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__max_iter": [100, 200, 500, 1000],
            "classifier__l1_ratio": [None],
        },
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100, 1000],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear", "saga"],
            "classifier__max_iter": [100, 200, 500, 1000],
            "classifier__l1_ratio": [None],
        },
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100, 1000],
            "classifier__penalty": ["elasticnet"],
            "classifier__solver": ["saga"],
            "classifier__max_iter": [100, 200, 500, 1000],
            "classifier__l1_ratio": [0.1, 0.5, 0.7, 0.9],
        },
    ]

    # # Prior Best found values
    # param_grid = {
    #     "classifier__C": [100],
    #     "classifier__l1_ratio": [None],
    #     "classifier__max_iter": [500],
    #     "classifier__penalty": ["l2"],
    #     "classifier__solver": ["lbfgs"],
    # }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        verbose=0,
        n_jobs=-1,
    )

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}")

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate the model
    y_pred = best_model.predict(X_test)
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test set accuracy: {test_accuracy}")

    # Predicted probabilities
    y_prob = (
        best_model.predict_proba(X_test)
        if hasattr(best_model, "predict_proba")
        else None
    )

    # Create classification report table
    report_table = classification_report(
        y_test, y_pred, target_names=["CN", "MCI", "AD"], output_dict=True
    )
    report_df = pd.DataFrame(report_table)
    plot_classification_report(report_df, f"log_reg_classification_report_year_{year}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, f"log_reg_confusion_matrix_year_{year}", year)  # type: ignore

    # Calculate overall feature importance
    coefficients = best_model.named_steps["classifier"].coef_

    # Take the absolute values of the coefficients and average them across all classes
    average_importance = coefficients.mean(axis=0)

    # Create a DataFrame for the overall feature importance
    importance_df = pd.DataFrame(
        average_importance, index=preprocessed_feature_names, columns=["Importance"]
    ).sort_values(by="Importance", ascending=False)

    # Calculate ROC
    roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") # type: ignore
    print(f"ROC: {roc_auc_ovr}")

    plot_feature_importance(
        importance_df, f"log_reg_feature_importance_year_{year}", year
    )

    metrics.update_metrics(
        run=year,
        y_true=y_test,
        y_pred=y_pred,  # type: ignore
        y_prob=y_prob,  # type: ignore
    )
