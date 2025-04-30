"""
Which metropolitan area is the one with the highest rate of identity tehfts per 
100,000 population. 
 
 - If you don't have their population size in 2023, 
   use two years where you know the censuses (or an estimate of the censurs) and 
   linearly interpolate between them to estimate the 2023 population size. 

 - Be sure to robustly match the names of metropolitan areas: 
   Use only the city and state portion of the name, ignoring suffixes like 
   "Metropolitan Statistical Area" or "MSA" and normalizing punctuation. Drop entries 
   where there's no match in the html for the areas fraud reports.

 - Ignore orderings in the csv files and focus on the numerical data presented.
"""

import pandas as pd
from data_utils import get_absolute_population_df, get_fraud_number_across_states
import re

df_reports = get_fraud_number_across_states()
df_population = get_absolute_population_df()

# Normalize names
def normalize_name(name):
    name = name.lower()
    name = re.sub(r'\s*(metropolitan statistical area|msa)$', '', name)
    name = re.sub(r'[^a-z0-9]+', '', name)
    return name

# Step 1: Normalize the metropolitan area names.
df_reports['msa_key'] = df_reports['Metropolitan Area'].apply(normalize_name)
df_population['msa_key'] = df_population['Metropolitan statistical area'].apply(normalize_name)

# Step 2: Merge on cleaned msa_key
df_merged = pd.merge(df_reports, df_population, on='msa_key', how='inner')

# Step 3: Calculate fraud rate per 100k
df_merged['fraud_per_100k'] = (df_merged['# of Reports'] / df_merged['2023 interpolated']) * 100_000

# Step 4: Find the metro with the highest rate
max_row = df_merged.loc[df_merged['fraud_per_100k'].idxmax()]

# Output
metro_name = max_row['Metropolitan Area']
print(metro_name)
