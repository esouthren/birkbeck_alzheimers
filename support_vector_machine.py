import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from model_helper_functions import *
from sklearn.svm import SVC
from plotting_helper_functions import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
)
from joblib import parallel_backend


def run_support_vector_machine_model(
    df: pd.DataFrame, year: int, metrics: ModelMetrics
) -> None:
    """
    Runs a Support Vector Machine (SVM) model with grid search for hyperparameter tuning, evaluates its performance, and updates metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the input data.
    year : int
        The year for which the model is being run.
    metrics : ModelMetrics
        An instance of the ModelMetrics class to store and update the metrics.

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
    ) = create_pipeline(df, model=SVC(probability=True))

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 0.1, 1],
        'classifier__degree': [3],
        'classifier__coef0': [0.0, 0.5]
    }


    # Prior best grid search
    # param_grid = {
    #     "classifier__C": [0.1, 1, 10, 100],
    #     "classifier__kernel": ["linear"],
    #     "classifier__degree": [2,3,4],
    #     "classifier__gamma": ["scale"],
    # }

    with parallel_backend("loky"):

        grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=100,  # number of parameter settings that are sampled
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
        grid_search.fit(X_train, y_train)

        # Fit the GridSearchCV object to the training data
        grid_search.fit(X_train, y_train)

        # Print the best parameters and the best score
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_}")

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate the model
        y_pred = best_model.predict(X_test) # type: ignore
        test_accuracy = best_model.score(X_test, y_test) # type: ignore
        print(f"Test set accuracy: {test_accuracy}")

        # Predicted probabilities
        y_prob = (
            best_model.predict_proba(X_test) # type: ignore
            if hasattr(best_model, "predict_proba")
            else None
        )

        # Print the classification report
        report_table = classification_report(
            y_test, y_pred, target_names=["CN", "MCI", "AD"], output_dict=True
        )
        report_df = pd.DataFrame(report_table).transpose()
        plot_classification_report(report_df, f"svm_classification_report_year_{year}")

        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, f"svm_confusion_matrix_year_{year}", year)  # type: ignore

        # Calculate ROC
        roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") # type: ignore
        print(f"ROC: {roc_auc_ovr}")

        result = permutation_importance(
            # TODO consider n_repeats 10
            best_model.named_steps["classifier"], # type: ignore
            x_train_preprocessed,
            y_train,
            n_repeats=5,
            random_state=42,
            scoring="accuracy",
        )

        importance_df = pd.DataFrame(result.importances_mean, index=preprocessed_feature_names, columns=["Importance"]).round(2)  # type: ignore
        plot_feature_importance(importance_df, f"svm_feature_importance_year_{year}", year)  # type: ignore

        metrics.update_metrics(
            run=year,
            y_true=y_test,
            y_pred=y_pred,  # type: ignore
            y_prob=y_prob,  # type: ignore
        )
