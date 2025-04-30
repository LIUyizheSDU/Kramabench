"""
Are the report counts of for "frauds and other data" in 2024 consistent for the Metropolitan area of Miami-Fort Lauderdale-West Palm Beach?
 -- State MSA Fraud And Other Data/Floria.csv
 -- CSN_Metropolitan_Areas_Fraud_and_Other_Reports.csv
 -- bool
 -- hard: Not utf-8 decodable.
 -- (answer: True)
"""

import re
from data_utils import read_clean_numeric_csv


df1 = read_clean_numeric_csv("../input/raw/CSVs/State MSA Fraud and Other data/Florida.csv", encoding="ISO-8859-1")
df2 = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Metropolitan_Areas_Fraud_and_Other_Reports.csv", encoding="ISO-8859-1")

# Normalize function to match similar area names
def normalize_area(name):
    return re.sub(r"[^\w]", "", name).lower()

# Normalize target area
target_area_1 = normalize_area("Miami-Fort Lauderdale-West Palm Beach, FL Metropolitan Statistical Area")
target_area_2 = normalize_area("Miami-Fort Lauderdale-West Palm Beach FL Metropolitan Statistical Area")

# Get row from df1
row1 = df1[df1["Metropolitan Area"].apply(normalize_area) == target_area_1]
# Get row from df2
row2 = df2[df2["Metropolitan Area"].apply(normalize_area) == target_area_2]

# Check and compare
if row1.empty:
    print(False)
elif row2.empty:
    print(False)
else:
    reports_1 = int(row1["# of Reports"].iloc[0])
    reports_2 = int(row2["# of Reports"].iloc[0])
    per_100k = int(row2["Reports per 100K Population"].iloc[0])
    
    print(reports_1 == reports_2)
