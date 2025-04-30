"""
What is the ratio of reported credit card frauds between 2024 and 2020? (2024 reports ) / (2020 reports)
 -- CSN_Top_Three_Identity_Theft_Reports_by_Year.csv
 -- number
 -- hard: Columns are named wrong
"""
from data_utils import read_clean_numeric_csv


df = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Top_Three_Identity_Theft_Reports_by_Year.csv")

# Normalize theft type strings
df["Year"] = df["Year"].str.strip().str.lower()

# Get values safely with corrected logic
def get_reports(theft_type, year):
    match = df[(df["Theft Type"] == year) & (df["Year"] == theft_type.lower())]
    if not match.empty:
        return match["# of Reports"].iloc[0]
    else:
        raise ValueError(f"No data found for theft type '{theft_type}' and year {year}")

try:
    cc_2020 = get_reports("Credit Card", 2020)
    cc_2024 = get_reports("Credit Card", 2024)
    ratio = cc_2024 / cc_2020
    print(ratio)
except ValueError as e:
    print(e)

