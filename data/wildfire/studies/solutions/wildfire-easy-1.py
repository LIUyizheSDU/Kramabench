import pandas as pd
import numpy as np
import geopandas as gpd

df = pd.read_csv("../../data/wildfire/input/noaa_wildfires_monthly_stats.csv", skiprows=3)

df = df.sort_values('Date')
    
# Create a function to check if dates are consecutive months
def are_consecutive(date1, date2):
    year1, month1 = divmod(date1, 100)
    year2, month2 = divmod(date2, 100)

    # Convert to total months
    total_months1 = year1 * 12 + month1
    total_months2 = year2 * 12 + month2

    return total_months2 - total_months1 == 1

max_acres = 0
max_period = None

# Iterate through possible 3-month periods
for i in range(len(df) - 2):
    # Check if months are consecutive
    if (are_consecutive(df['Date'].iloc[i], df['Date'].iloc[i+1]) and 
        are_consecutive(df['Date'].iloc[i+1], df['Date'].iloc[i+2])):

        # Calculate total acres for this period
        period_acres = df['Acres Burned'].iloc[i:i+3].sum()

        # Update maximum if this period is larger
        if period_acres > max_acres:
            max_acres = period_acres
            max_period = df.iloc[i:i+3]

print({
    'Start Date': max_period['Date'].iloc[0],
    'End Date': max_period['Date'].iloc[-1],
    'Total Acres Burned': max_acres,
    'Period Data': max_period
})