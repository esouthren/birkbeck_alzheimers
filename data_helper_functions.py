import pandas as pd
from pandas import DataFrame
import numpy as np
from datetime import datetime
from typing import Optional


def write_csv(path: str, df: DataFrame) -> None:
    """
    Writes a data frame to a CSV file.

    Parameters
    ----------
    path : String
        File path to write to.
    df : pandas.DataFrame
        The data to write.

    Returns
    -------
    None
    """
    output_file_path = f"data/{path}"
    df.to_csv(output_file_path, index=False)
    print(f"Wrote to {path}! Rows: {df.iloc[:, 0].count()}")


def print_value_counts(df: DataFrame, cols: list[str]) -> None:
    """
    Prints the value counts for specified columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    cols : list of str
        The list of columns for which to print value counts.

    Returns
    -------
    None
    """
    for column in cols:
        print(f"Value counts for column '{column}':")
        data = df[column].value_counts()
        sorted_dict = dict(sorted(data.items(), key=lambda item: item[0]))  # type: ignore
        for key, value in sorted_dict.items():
            print(f"{key}: {value}")


def print_value_counts_of_diagnosis(df: DataFrame, cols: list[str]) -> None:
    """
    Prints the value counts of specified columns in the DataFrame,
    cross-tabulated by the 'DIAGNOSIS' column. 'DIAGNOSIS' has three
    value options.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    cols : list of str
        The list of column names to print the value counts of.

    Returns
    -------
    None
    """
    for column in cols:
        print(f"\nValue counts for column '{column}':")

        # Create a crosstab for the value counts by diagnosis
        crosstab = pd.crosstab(df[column], df["DIAGNOSIS"], dropna=False)

        # Sort the crosstab by index
        crosstab = crosstab.sort_index()

        # Print the crosstab in a formatted table
        print(crosstab.to_string())


def calculate_age_at_date(dob: str, date: datetime) -> Optional[float]:
    """
    Calculates the age in years based on the date of birth and a given date.

    Parameters
    ----------
    dob : str
        The date of birth, in 'MM/YYYY' format.
    date : datetime
        The date from which to calculate the age.

    Returns
    -------
    float
        The calculated age in years. Returns None if the date of birth is not valid.
    """
    if not dob:
        return None

    try:
        dob_month, dob_year = map(int, dob.split("/"))
    except ValueError:
        return None  # type: ignore

    # Calculate the difference in years
    age_years = date.year - dob_year

    # Adjust the age based on the month
    _, months_passed = divmod(date.month - dob_month, 12)

    if date.month < dob_month or (date.month == dob_month and date.day < 1):
        age_years -= 1
        months_passed = 11 - (dob_month - date.month - 1)

    # Calculate the fraction of the year as months/12
    age = age_years + (months_passed / 12.0)

    return age


def parse_demographics_data(csv_path: str) -> pd.DataFrame:
    """
    Parses raw Demographics ADNI data and prepares it for merging with other data sources.

    Used Data fields:
    PTID : str
        Patient Identifier
    PTGENDER : int
        1=male, 2=female
    PTDOB : str
        Patient date of birth
    PTMARRY : int
        1=Married; 2=Widowed; 3=Divorced; 4=Never married; 5=Unknown
    PTEDUCAT : int
        0..20
    PTHOME : int
        1=House; 2=Condo/Co-op (owned); 3=Apartment (rented); 4=Mobile Home;
        5=Retirement Community; 6=Assisted Living; 7=Skilled Nursing Facility; 8=Other (specify)
    PTRACCAT : int
        1=American Indian or Alaskan Native; 2=Asian; 3=Native Hawaiian or Other Pacific Islander;
        4=Black or African American; 5=White; 6=More than one race; 7=Unknown

    Parameters
    ----------
    csv_path : str
        The file path to the CSV file containing the raw demographics data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the cleaned and prepared demographics data.
    """
    df = pd.read_csv(csv_path, low_memory=True)
    cols = ["PTID", "PTGENDER", "PTDOB", "PTMARRY", "PTEDUCAT", "PTHOME", "PTRACCAT"]
    df = df[cols]

    # Drop duplicates
    df = df.drop_duplicates(subset=["PTID"], keep="first")

    # Replace negative or 0 values
    cols = [
        "PTGENDER",
        "PTMARRY",
        "PTEDUCAT",
        "PTHOME",
        "PTRACCAT",
    ]
    df[cols] = df[cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df[cols] = df[cols].where(df[cols] > 0, np.nan)

    write_csv("demographics_scrubbed.csv", df)

    num_unique_values = df['PTID'].nunique()
    print(f'Number of unique values in demographics data: {num_unique_values}')

    return df

def parse_battery_tests_data(csv_path):
    """
    Parses raw battery tests ADNI data and prepares it for analysis.

    Parameters
    ----------
    csv_path : str
        The file path to the CSV file containing the raw battery tests data.

    Used Data fields from frame:
    PTID : str
        Unique patient ID
    VISDATE : str
        Date of administered test

    ** Logical Memory Test **
    LIMMTOTAL : int
        Number of story units recalled immediately. [0..25]
    LDELTOTAL : int
        Number of story units recalled after delay [0..25]

    ** AVLT (Auditory Verbal Learning Test) **
    AVTOT1: int
        Average total recall from List A in first trial [0..15]
    AVERR1 : int
        Number of intrusions to List A in first trial (errors). [0..?]
    AVTOT5: int
        Average total recall from List A in fifth trial [0..15]
    AVERR5 : int
        Number of intrusions to List A in fifth trial (errors). [0..?]
    AVTOTB : int
        Average total recall from List B. [0..15]
    AVERRB : int
        Number of intrusions to List B (errors). [0..?]

    ** Clock Drawing Test **
    CLOCKSCOR : int
        Clock drawing accuracy score. [0..5], 5 = 'best' performance
    COPYSCOR : int
        Clock copying accuracy score. [0..5], 5 = 'best' performance

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the cleaned and prepared battery tests data.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    cols = [
        "PTID",
        "VISDATE",
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

    # Remove negative values
    df = df[cols].replace(-1, np.nan)

    # Cast VISDATE to DateTime
    df["VISDATE"] = pd.to_datetime(df["VISDATE"])

    write_csv("battery_tests_scrubbed.csv", df)
    num_unique_values = df['PTID'].nunique()
    print(f'Number of unique values in battery tests: {num_unique_values}')

    return df


def parse_diagnosis_data(csv_path: str) -> pd.DataFrame:
    """
    Parses raw diagnosis data and prepares it for analysis.

    Data fields used from frame:
    PTID : str
        Patient identifier
    DIAGNOSIS : str
        Diagnosis of the patient. [0,1,2]
    EXAMDATE : str
        Date of the examination

    Parameters
    ----------
    csv_path : str
        The file path to the CSV file containing the raw diagnosis data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the cleaned and prepared diagnosis data.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.sort_values(by=["PTID", "EXAMDATE"], ascending=[True, False])
    df = df[["PTID", "DIAGNOSIS", "EXAMDATE"]]
    df.rename(columns={"EXAMDATE": "EXAM_DATE"}, inplace=True)

    # Drop any row with no value for diagnosis
    df = df.dropna(subset=["DIAGNOSIS", "EXAM_DATE"])
    df["EXAM_DATE"] = pd.to_datetime(df["EXAM_DATE"], errors="coerce")

    # Write file
    write_csv("diagnosis_scrubbed.csv", df)

    num_unique_values = df['PTID'].nunique()
    print(f'Number of unique values in diagnosis: {num_unique_values}')

    return df
