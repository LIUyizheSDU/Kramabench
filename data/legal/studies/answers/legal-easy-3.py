"""
Give the ratio of identity theft reports in 2024 vs 2001?
 -- easy
 -- number
 -- 2024_CSN_Number_of_Reports_by_Type.csv
"""
from data_utils import read_clean_numeric_csv


df = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Number_of_Reports_by_Type.csv")

# Extract Identity Theft values
id_theft_2001 = df.loc[df["Year"] == 2001, "Identity Theft "].values[0]
id_theft_2024 = df.loc[df["Year"] == 2024, "Identity Theft "].values[0]

# Compute ratio
ratio = id_theft_2024 / id_theft_2001

print(ratio)
