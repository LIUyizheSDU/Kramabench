import pandas as pd
import os

aqi_df = pd.read_csv('../../data/wildfire/input/annual_aqi_by_county_2024.csv')
aqi_df['good day proportion'] = aqi_df['Good Days'] / aqi_df['Days with AQI']
aqi_df_state = aqi_df.groupby(['State']).mean(['good day proportion']).reset_index()
df_combined = pd.read_csv('../../data/wildfire/input/Wildfire_Acres_by_State.csv')
df_combined['acres per capita'] = df_combined['Total Acres Burned'] / df_combined['Population']

augmented_df = aqi_df_state.merge(df_combined[['State', 'Total Acres Burned', 'Population']], on='State', how='left')
# Ensure the columns are numeric
augmented_df['good day proportion'] = pd.to_numeric(augmented_df['good day proportion'], errors='coerce')
augmented_df['Total Acres Burned'] = pd.to_numeric(augmented_df['Total Acres Burned'], errors='coerce')

# Calculate the correlation
correlation = augmented_df[['good day proportion', 'Total Acres Burned']].corr().iloc[0, 1]
print(f"Correlation between good day proportion and total acres burned: {correlation}")
augmented_df['bad days'] = augmented_df['Unhealthy Days'] + augmented_df['Very Unhealthy Days'] + augmented_df['Hazardous Days']
# Drop 'County' and 'State' columns for correlation calculation
augmented_df = augmented_df.drop(columns=['State'])
augmented_df.corr()