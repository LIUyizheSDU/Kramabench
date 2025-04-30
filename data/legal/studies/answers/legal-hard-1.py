"""
Report the average number of reported identity thefts for all metropolitan areas 
that are larger than one million in population in 2023. 

 - If you don't have their population size in 2023, 
   use two years where you know the censuses (or an estimate of the censurs) and 
   linearly interpolate between them to estimate the 2023 population size. 

 - Be sure to robustly match the names of metropolitan areas: 
   Use only the city and state portion of the name, ignoring suffixes like 
   "Metropolitan Statistical Area" or "MSA" and normalizing punctuation. Drop entries 
   where there's no match in the html for the areas fraud reports.

 --- hard, 
 --- number
 --- html, State MSA Identitiy Theft Data/ *
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

# Normalize the metropolitan area names.
df_reports['msa_key'] = df_reports['Metropolitan Area'].apply(normalize_name)
df_population['msa_key'] = df_population['Metropolitan statistical area'].apply(normalize_name)

# Merge on msa_key.
df_merged = pd.merge(df_reports, df_population, on='msa_key', how='inner')

# Filter for MSAs with >1M population
df_large = df_merged[df_merged['2023 interpolated'] > 1_000_000]

# Compute average number of reports per 100K
avg_reports_per_100k = df_large['# of Reports'].mean()

print(avg_reports_per_100k)
