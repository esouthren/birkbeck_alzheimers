import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from model_helper_functions import *
from plotting_helper_functions import (
    plot_feature_importance,
    plot_classification_report,
    plot_confusion_matrix,
)


def run_random_forest_model(df: pd.DataFrame, year: int, metrics: ModelMetrics) -> None:
    """
    Trains and evaluates a Random Forest model using the given data and year for reference.

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
    ) = create_pipeline(df, model=RandomForestClassifier(random_state=42))

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1,
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

    # Print the classification report
    # print(f'Classification Report for Random Forest, Year {year}')
    # print(classification_report(y_test, y_pred))

    # Create classification report table
    report_table = classification_report(
        y_test, y_pred, target_names=["CN", "MCI", "AD"], output_dict=True
    )
    report_df = pd.DataFrame(report_table).transpose()
    plot_classification_report(
        report_df, f"random_forest_classification_report_year_{year}"
    )

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, f"random_forest_confusion_matrix_year_{year}", year)  # type: ignore

    # Display feature importance
    importances = best_model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame(
        importances, index=preprocessed_feature_names, columns=["Importance"]
    )
    plot_feature_importance(importance_df, f"random_forest_feature_importance_year_{year}", year)  # type: ignore

    # Calculate ROC
    roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") # type: ignore
    print(f"ROC: {roc_auc_ovr}")
    
    metrics.update_metrics(
        run=year,
        y_true=y_test,
        y_pred=y_pred,  # type: ignore
        y_prob=y_prob,  # type: ignore
    )
