from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from typing import Union, List
import numpy as np
import math
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.lines import Line2D

# Plot-wide style constants
n_cols = 4
title_fontsize = 16
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": title_fontsize,
    }
)
legend_map = {
    1.0: "Cognitively Normal",
    2.0: "Mild Impairment",
    3.0: "Alzheimer's Diagnosis",
}
sns.set_palette("pastel")
sns.set_theme(style="whitegrid", font="Times New Roman")


def plot_confusion_matrix(
    y_test: Union[List[int], List[str], np.ndarray],
    y_pred: Union[List[int], List[str], np.ndarray],
    filename: str,
    year_window: int,
    show_plot: bool = False,
) -> None:
    """
    Plots and saves the confusion matrix for the given true and predicted labels.

    Parameters
    ----------
    y_test : Union[List[int], List[str], np.ndarray]
        True labels.
    y_pred : Union[List[int], List[str], np.ndarray]
        Predicted labels.
    filename : str
        The filename to save the plot as.
    year_window: str
        Time position of the chart
    show_plot : bool, optional
        If True, display the plot. Default is False.

    Returns
    -------
    None
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 14},
    )

    plt.title(
        f"Confusion Matrix for Model Predicting Diagnosis \nOutcomes {year_window - 1}-{year_window} Years Before Examination",
        fontsize=title_fontsize,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(ticks=np.arange(len(conf_matrix)) + 0.5, labels=np.arange(len(conf_matrix)))  # type: ignore
    plt.yticks(ticks=np.arange(len(conf_matrix)) + 0.5, labels=np.arange(len(conf_matrix)), rotation=0)  # type: ignore

    if show_plot:
        plt.show()

    save_plot_and_maybe_show(plt, filename, show_plot)


def plot_feature_importance(
    df: pd.DataFrame,
    filename: str,
    year_window: int,
    show_plot: bool = False,
) -> None:
    """
    Plots and saves the feature importance from a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing feature importance data.
    filename : str
        The filename to save the plot as.
    year_window: str
        Time position of the chart
    show_plot : bool, optional
        If True, display the plot. Default is False.

    Returns
    -------
    None
    """
    top_10_importances = df.nlargest(10, "Importance").reset_index()
    top_10_importances.columns = ["Feature", "Importance"]
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x="Importance", y="Feature", data=top_10_importances)

    tick_positions = range(len(top_10_importances.index))
    ax.yaxis.set_major_locator(FixedLocator(tick_positions))
    ax.set_yticklabels([format_name(label) for label in top_10_importances["Feature"]])

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance in Model Predicting Diagnosis Outcomes\n{year_window -1}-{year_window} Years Before Examination",
        fontsize=title_fontsize,
    )
    plt.tight_layout()

    save_plot_and_maybe_show(plt, filename, show_plot)


def plot_histogram_distribution(
    df: pd.DataFrame, headers: List[str], year_window: int, show_plot: bool = False
):
    """
    Plots histograms of distributions of data for specified columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    headers : List[str]
        A list of column names to plot histograms for.
    year_window: str
        Time position of the chart
    show_plot : bool, optional
        If True, display the plots. Default is False.

    Returns
    -------
    None
    """
    num_headers = len(headers)
    n_rows = math.ceil(num_headers / n_cols)

    _, axs = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axs = axs.flatten()

    for i, header in enumerate(headers):
        formatted_name = format_name(header)
        sns.histplot(df[header], bins="auto", edgecolor="k", ax=axs[i], kde=False, alpha=0.7)  # type: ignore
        axs[i].set_title(formatted_name)
        axs[i].set_xlabel(formatted_name)
        axs[i].set_ylabel("Frequency")
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.suptitle(
        f"Feature Distribution {year_window -1}-{year_window} Years Prior to Diagnosis Examination"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore # Adjust layout to prevent overlap with super title

    save_plot_and_maybe_show(
        plt, f"histogram_features_distribution_in_year_{year_window}", show_plot
    )


def plot_histogram_diagnosis_target_against_feature(
    df, headers, year_window: int, show_plot: bool = False
):
    """
    Plots histograms of the distributions of specified features against diagnosis values in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    headers : List[str]
        A list of column names to plot histograms for.
    year_window: str
        Time position of the chart
    show_plot : bool, optional
        If True, display the plots. Default is False.

    Returns
    -------
    None
    """
    num_headers = len(headers)
    rows = math.ceil(num_headers / n_cols)

    diagnosis_values = sorted(df["DIAGNOSIS"].unique())
    fig = plt.figure(figsize=(3 * n_cols, 3 * rows))

    # Initialize lists to collect handles and labels for the legend
    handles = []
    labels = []

    for i, header in enumerate(headers):
        formatted_name = format_name(header)
        plt.subplot(rows, n_cols, i + 1)
        for diagnosis in diagnosis_values:
            subset = df[df["DIAGNOSIS"] == diagnosis]
            hist = sns.histplot(
                subset[header],
                bins="auto",
                kde=False,
                label=legend_map[diagnosis],
                element="step",
                common_norm=False,
                multiple="dodge",
                alpha=0.4,
            )

            if i == 0 and diagnosis == diagnosis_values[0]:
                handle, label = hist.get_legend_handles_labels()
                handles.append(handle[0])
                labels.append(legend_map[diagnosis])

        plt.title(formatted_name)
        plt.xlabel(formatted_name)
        plt.ylabel("Frequency")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Line2D([0], [0], color=color, lw=4, label=legend_map[diagnosis])
        for diagnosis, color in zip(
            diagnosis_values, sns.color_palette("muted", len(diagnosis_values))
        )
    ]
    fig.legend(
        handles,
        legend_map.values(),
        loc="upper center",
        ncol=len(legend_map),
        bbox_to_anchor=(0.5, 0.96),  # Adjust this value to move the legend lower
        bbox_transform=fig.transFigure,
    )

    plt.suptitle(
        f"Feature Distributions Per Diagnosis Target {year_window -1}-{year_window} Years Prior to Diagnosis Examination",
        y=1,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    save_plot_and_maybe_show(
        plt,
        f"histogram_diagnosis_against_features_in_year_{year_window}",
        show_plot,
    )


def plot_box_diagnosis_against_feature(df, headers, year, show_plot=False):
    cols = 4
    num_headers = len(headers)
    rows = math.ceil(num_headers / cols)

    # Set up the matplotlib figure
    plt.figure(figsize=(3 * cols, 3 * rows))

    for i, header in enumerate(headers):
        formatted_name = format_name(header)

        plt.subplot(rows, cols, i + 1)
        ax = sns.boxplot(x="DIAGNOSIS", y=header, data=df)
        plt.title(f"{formatted_name}")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel(formatted_name)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    plt.suptitle(
        f"Feature Distributions Per Diagnosis Target {year -1}-{year} Years Prior to Diagnosis Examination",
        y=1,
    )

    save_plot_and_maybe_show(
        plt,
        f"histogram_diagnosis_against_features_in_year_{year}",
        show_plot,
    )


def plot_classification_report(
    df: pd.DataFrame, filename: str, show_plot: bool = False
) -> None:
    """
    Plots table of model classiciation report.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    filename : List[str]
        The name for the PNG file.
    show_plot : bool, optional
        If True, display the plots. Default is False.

    Returns
    -------
    None
    """
    # Round values to 2 DP
    df = pd.DataFrame(df).transpose().round(2)

    fig, ax = plt.subplots(figsize=(12, 10))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )

    # Customize table font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2)  # Adjust the table scale for better readability

    fig.text(
        0.5,
        0.65,
        "Classification Report",
        ha="center",
        fontsize=title_fontsize,
    )

    plt.tight_layout()

    save_plot_and_maybe_show(plt, filename, show_plot)


def plot_all_model_metrics(model_metrics):
    metric_names = list(model_metrics.metrics.keys())

    _, axs = plt.subplots(2, 3, figsize=(10, 8))
    axs = axs.flatten()

    tick_labels = ["0-1", "1-2", "2-3"]

    for i, metric_name in enumerate(metric_names):
        df = model_metrics.metric_to_dataframe(metric_name)
        df.index -= 1
        ax = axs[i]
        df.plot(ax=ax, kind="line", marker="o")
        ax.set_title(metric_name.capitalize())
        ax.set_xlabel("Year Prior Window")
        ax.set_ylabel("%")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(tick_labels)
        ax.legend().remove()
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])  # type: ignore
    plt.suptitle(
        f"{model_metrics.name} Model Metrics For All Runs",
        y=1,
    )
    name = f'{(model_metrics.name).lower().replace(" ", "_")}_all_metrics'
    save_plot_and_maybe_show(plt, name, False)


def format_name(str: str) -> str:
    """
    Formats a "SCREAMING_CAPS" string to "Screaming Caps".
    Each word is capitalized, numeric parts are kept as is.

    Parameters
    ----------
    str : str
        The column header to format.

    Returns
    -------
    str
    """
    words = str.split("_")
    formatted_words = []
    for word in words:
        if word.replace(
            ".", "", 1
        ).isdigit():  # Check if the word is numeric, allowing for decimal points
            formatted_words.append(word)
        else:
            formatted_words.append(word.capitalize())

    return " ".join(formatted_words)


def save_plot_and_maybe_show(plt, filename: str, show_plot: bool) -> None:
    """
    Saves a plot to a file and optionally displays it.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The plot to be saved and optionally shown.
    filename : str
        The filename for saving the plot.
    show_plot : bool
        If True, the plot will be displayed.

    Returns
    -------
    None
    """
    if show_plot:
        plt.show()
    path = f"figures/{filename}.png"
    plt.savefig(path, format="png")
    print(f"Saved plot: {path}")
    plt.close("all")
