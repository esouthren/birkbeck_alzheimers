import pandas as pd
import numpy as np
from data_helper_functions import *
from plotting_helper_functions import (
    plot_box_diagnosis_against_feature,
    plot_histogram_diagnosis_target_against_feature,
    plot_histogram_distribution,
)


def prepare_model_data(year_ago_window: int) -> pd.DataFrame:
    """
    Prepares ADNI data to be ingested by training models.

    In summary:
    - Takes Demographics data, removes duplicates.
    - Takes Battery Test data, and retrieves several tests.
    - Limits test data to tests taken within a specified year window of the diagnosis.
    - Averages results across all tests in that year.
    - Merges demographics, tests, and diagnosis data.
    - Calculates age based on date at time of diagnosis.

    Missing data is handled after the training/testing split to prevent data leakage.

    Parameters
    ----------
    year_ago_window : int
        The number of years prior to the diagnosis to consider for test data.
        This represents a 'sliding window', i.e if the value is 3, all tests taken
        between 2-3 years before the diagnosis exam will be included and averaged.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prepared data.
    """
    demographics_df = parse_demographics_data("data/original/demographics.csv")
    tests_df = parse_battery_tests_data("data/original/battery_tests.csv")
    diagnosis_df = parse_diagnosis_data("data/original/diagnosis_summary_full.csv")

    # Convert dates to datetime format
    diagnosis_df["EXAM_DATE"] = pd.to_datetime(diagnosis_df["EXAM_DATE"])
    tests_df["VISDATE"] = pd.to_datetime(tests_df["VISDATE"])

    # Prepare the output dataframe
    output_rows = []
    output_columns = [
        "PTID",
        "EXAM_DATE",
        "DIAGNOSIS",
        "LIMMTOTAL",
        "LDELTOTAL",
        "AVTOT1",
        "AVERR1",
        "AVTOT5",
        "AVERR5",
        "AVTOTB",
        "AVERRB",
        "CLOCKSCOR",
        "COPYSCOR",
    ]

    for _, row in (diagnosis_df).iterrows():
        id = row["PTID"]
        exam_date = row["EXAM_DATE"]
        diagnosis = row["DIAGNOSIS"]

        # Calculate the start and end dates for the window
        start_date = exam_date - pd.DateOffset(years=year_ago_window)
        end_date = exam_date - pd.DateOffset(years=year_ago_window - 1)

        # Filter rows for the same PTID and within the specified window
        window_df = tests_df[
            (tests_df["PTID"] == id)
            & (tests_df["VISDATE"] >= start_date)
            & (tests_df["VISDATE"] < end_date)
        ]

        if not window_df.empty:
            # Calculate the mean of the columns
            avg_values = window_df[
                [
                    "LIMMTOTAL",
                    "LDELTOTAL",
                    "AVTOT1",
                    "AVERR1",
                    "AVTOT5",
                    "AVERR5",
                    "AVTOTB",
                    "AVERRB",
                    "CLOCKSCOR",
                    "COPYSCOR",
                ]
            ].mean()

            # Create a new row for the output dataframe
            new_row = {
                "PTID": id,
                "EXAM_DATE": exam_date,
                "DIAGNOSIS": diagnosis,
                "LIMMTOTAL": avg_values["LIMMTOTAL"],
                "LDELTOTAL": avg_values["LDELTOTAL"],
                "AVTOT1": avg_values["AVTOT1"],
                "AVERR1": avg_values["AVERR1"],
                "AVTOT5": avg_values["AVTOT5"],
                "AVERR5": avg_values["AVERR5"],
                "AVTOTB": avg_values["AVTOTB"],
                "AVERRB": avg_values["AVERRB"],
                "CLOCKSCOR": avg_values["CLOCKSCOR"],
                "COPYSCOR": avg_values["COPYSCOR"],
            }

            output_rows.append(new_row)

    output_df = pd.DataFrame(output_rows, columns=output_columns)

    # Merge in demographics data on shared PTID
    demographics_df = pd.read_csv("data/demographics_scrubbed.csv", low_memory=False)
    filtered_df = pd.merge(output_df, demographics_df, on="PTID", how="left")

    # Calculate age based on date of birth and EXAM_DATE
    filtered_df["AGE"] = filtered_df.apply(lambda row: calculate_age_at_date(row["PTDOB"], row["EXAM_DATE"]), axis=1)  # type: ignore
    filtered_df = filtered_df.drop(columns=["PTDOB"])

    #  Remove all values below 0
    features = [
        "LIMMTOTAL",
        "LDELTOTAL",
        "AVTOT1",
        "AVERR1",
        "AVTOT5",
        "AVERR5",
        "AVTOTB",
        "AVERRB",
        "CLOCKSCOR",
        "COPYSCOR",
        "PTEDUCAT",
        "AGE",
        "PTRACCAT",
        "PTMARRY",
    ]
    filtered_df[features] = (
        filtered_df[features].mask(filtered_df[features] < 0, np.nan)
    ).round(2)

    # Drop rows with large test data missing
    subset = [
        ["AVTOTB", "AVERRB", "AVERR1", "AVTOT5", "AVERR5", "AVTOTB"],
        ["CLOCKSCOR", "COPYSCOR"],
        ["LIMMTOTAL", "LDELTOTAL"],
    ]
    for pair in subset:
        filtered_df = filtered_df.dropna(subset=pair, how="all")

    filtered_df.rename(
        columns={
            "PTID": "ID",
            "LIMMTOTAL": "AVG_RECALL_IMMEDIATE_TOTAL",
            "LDELTOTAL": "AVG_RECALL_DELAYED_TOTAL",
            "AVTOT1": "AVG_RAV_LISTA_1_RECALL",
            "AVERR1": "AVG_RAV_LISTA_1_ERRORS",
            "AVTOT5": "AVG_RAV_LISTA_5_RECALL",
            "AVERR5": "AVG_RAV_LISTA_5_ERRORS",
            "AVTOTB": "AVG_RAV_LISTB_RECALL",
            "AVERRB": "AVG_RAV_LISTB_ERRORS",
            "CLOCKSCOR": "AVG_CLOCK_SCORE",
            "COPYSCOR": "AVG_CLOCK_COPY",
            "PTGENDER": "GENDER",
            "PTMARRY": "MARRIAGE_STATUS",
            "PTEDUCAT": "EDUCATION",
            "PTHOME": "HOME_TYPE",
            "PTRACCAT": "ETHNICITY",
        },
        inplace=True,
    )

    filtered_headers = list(
        set(filtered_df.columns) - set(["DIAGNOSIS", "ID", "EXAM_DATE"])
    )

    plot_histogram_diagnosis_target_against_feature(
        filtered_df, filtered_headers, year_ago_window
    )
    plot_histogram_distribution(filtered_df, filtered_headers, year_ago_window)
    plot_box_diagnosis_against_feature(filtered_df, filtered_headers, year_ago_window)

    filtered_df.to_csv(
        f"data/prepared_data_year_window_{year_ago_window}.csv", index=False
    )
    return filtered_df
