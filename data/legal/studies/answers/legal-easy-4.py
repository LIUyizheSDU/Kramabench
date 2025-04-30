"""
How many frauds were reported by FTC over the web between 2022 and 2024 in total.
 -- easy
 -- number
 -- 2024_Data_Contributors.csv
"""
from data_utils import read_clean_numeric_csv


df = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Data_Contributors.csv")

mask = (
    df["Year"].between(2022, 2024) &
    df["Data Contributor"].str.contains("FTC - Web Reports \(Fraud & Other\)", regex=True)
)

# Sum the relevant reports
total_fraud_web_reports = df.loc[mask, "# of Reports"].sum()
print(total_fraud_web_reports)

