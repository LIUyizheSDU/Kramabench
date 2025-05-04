import pandas as pd
import os

df_combined = pd.read_csv('../../data/wildfire/input/Wildfire_Acres_by_State.csv')
df_combined['acres per capita'] = df_combined['Total Acres Burned'] / df_combined['Population']

democrat_or_republican = pd.read_csv('../../data/wildfire/input/democratic_vs_republican_votes_by_usa_state_2020.csv')
democrat_or_republican['party'] = democrat_or_republican['percent_democrat'].apply(lambda x: 'democrat' if x >= 50 else 'republican')

df_combined = df_combined.merge(pd.DataFrame({
    'State': democrat_or_republican['state'],
    'party': democrat_or_republican['party'],
}), on='State', how='outer')

df = pd.read_csv("../../data/wildfire/input/wildfire_total_fires_p45_54.csv")
df_combined = df_combined.merge(df, on='State', how='outer')
df_combined.groupby('party').agg({'Total Fires': 'sum', 'Total Acres Burned': 'sum', 'Population': 'sum'}).reset_index()