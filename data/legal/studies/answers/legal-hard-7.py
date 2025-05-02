"""
Which fraud category was growing the fastest between 2020 and 2024 in relative terms?
 -- CSN_Top_Three_Identity_Theft_Reports_by_Year.csv
 -- hard: column names switched
"""
from data_utils import read_clean_numeric_csv

df = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Top_Three_Identity_Theft_Reports_by_Year.csv")

# Normalize category strings
df["Year"] = df["Year"].str.strip().str.lower()

# Pivot the data to have one row per category, columns = [2020, 2024]
pivot = df[df["Theft Type"].isin([2020, 2024])].pivot_table(
    index="Year",
    columns="Theft Type",
    values="# of Reports"
)

# Drop categories not present in both years
pivot = pivot.dropna()

# Compute relative growth
pivot["growth_ratio"] = pivot[2024] / pivot[2020]

# Find the max
fastest_growing = pivot["growth_ratio"].idxmax()
ratio = pivot["growth_ratio"].max()

print(fastest_growing.title())
