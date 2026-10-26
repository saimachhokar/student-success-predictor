"""Print a basic overview of the mathematics and Portuguese student datasets."""

from pathlib import Path

import pandas as pd


# Find the data folder relative to this script so it works from any directory.
DATA_DIR = Path(__file__).resolve().parent / "data"


def explore_dataset(filename):
    """Load one CSV file and print useful information about its contents."""
    # These CSV files separate values with semicolons instead of commas.
    # Reading a CSV loads it into memory and does not change the original file.
    data = pd.read_csv(DATA_DIR / filename, sep=";")

    print(f"\n{'=' * 60}\nDataset: {filename}\n{'=' * 60}")

    # Shape gives the number of rows (students) and columns (variables).
    print("\nShape (rows, columns):", data.shape)

    # head() shows the first five rows as a quick preview.
    print("\nFirst five rows:")
    print(data.head().to_string())

    # Column names tell us which variables are available in the dataset.
    print("\nColumn names:")
    print(data.columns.tolist())

    # isna() identifies missing values; sum() counts them for each column.
    print("\nMissing-value counts:")
    print(data.isna().sum().to_string())

    # describe() summarizes the data. include="all" includes text columns too.
    # Numeric summaries include the mean, minimum, maximum, and quartiles.
    # Text summaries include unique values and the most frequent value.
    # NaN in this summary can mean a statistic does not apply to that column.
    print("\nSummary statistics:")
    print(data.describe(include="all").to_string())


# Run the exploration only when this file is executed directly.
if __name__ == "__main__":
    for dataset_filename in ("student-mat.csv", "student-por.csv"):
        explore_dataset(dataset_filename)
