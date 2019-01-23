from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sys
import os
import pprint

pp = pprint.PrettyPrinter()

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]

    # pp.pprint(oecd_bli)
    # print(oecd_bli)
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # pp.pprint(oecd_bli)
    # print(oecd_bli)
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


data_path = os.path.join('datasets', 'lifesat', '')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

oecd_bli = pd.read_csv(data_path + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(data_path + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# print(type(oecd_bli))
# print(oecd_bli)
# print(gdp_per_capita)

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
