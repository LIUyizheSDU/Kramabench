#!/usr/bin/env python
# coding: utf-8
import pandas as pd
data_path = "../input"

city_path = "{}/worldcities.csv".format(data_path)
df = pd.read_csv(city_path)

countries = df.groupby("country")["population"].mean()
index = countries.idxmax()
print(index)



