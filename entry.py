from data_preparation import prepare_model_data
from logistic_regression import run_logistic_regression_model
from plotting_helper_functions import plot_all_model_metrics
from random_forest import run_random_forest_model
from significance_helper_functions import significance_testing
from support_vector_machine import run_support_vector_machine_model
from model_helper_functions import ModelMetrics

logistic_regression_metrics = ModelMetrics("Logistic Regression")
random_forest_metrics = ModelMetrics("Random Forest")
svm_metrics = ModelMetrics("SVM")

# For the windows 'within 1 year, within 1-2 years, and within 2-3 years ago'
dfs = []
for year_window in [1,2,3]:
    df = prepare_model_data(year_window)
    dfs.append(df)
    # print(f"Number of rows in df for year {year_window}: {len(df)}")
    # significance_testing(df)

    # run_logistic_regression_model(df, year_window, logistic_regression_metrics)
    # run_random_forest_model(df, year_window, random_forest_metrics)
    run_support_vector_machine_model(df, year_window, svm_metrics)

# plot_all_model_metrics(logistic_regression_metrics)
# plot_all_model_metrics(random_forest_metrics)
# plot_all_model_metrics(svm_metrics)
