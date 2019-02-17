import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# pp = pprint.PrettyPrinter()

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]

    # pp.pprint(oecd_bli)
    # print(oecd_bli)
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # pp.pprint(oecd_bli)
    # print(oecd_bli)
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    # print(gdp_per_capita)
    gdp_per_capita.set_index("Country", inplace=True)
    # print(gdp_per_capita)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


def main_1_3_3():
    data_path = os.path.join('datasets', 'lifesat', '')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 250)
    oecd_bli = pd.read_csv(data_path + "oecd_bli_2015.csv", thousands=',')
    gdp_per_capita = pd.read_csv(data_path + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1',
                                 na_values="n/a")
    # print(type(oecd_bli))
    # print(oecd_bli)
    # print(gdp_per_capita)
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    # print(country_stats)
    # print(country_stats.shape)

    # country_stats DataFrame에서  GDP per capita Series하나를 분리 해서 np.c_로 array로 만든다.
    X = np.c_[country_stats['GDP per capita']]
    y = np.c_[country_stats['Life satisfaction']]

    country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
    plt.show()

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    X_new = [[22587]]
    print(model.predict(X_new))






# ===============================================================================================
# 2.3.2
# ===============================================================================================
import tarfile
from six.moves import urllib
from pathlib import Path
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    # csv_path = os.path.join(housing_path, 'housing.csv')
    csv_path = Path('./datasets/housing') / 'housing.csv'
    return pd.read_csv(csv_path)


def main_2_3_3() :
    housing = load_housing_data()
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 200)
    print(housing.head())
    print(housing.info())

    vc = housing['ocean_proximity'].value_counts()
    print(vc.index)
    print(vc)

    h_desc = housing.describe()
    print(h_desc)

    housing.hist(bins=50, figsize=(20, 15))
    plt.show()










if __name__ == '__main__':
    # main_1_3_3()
    main_2_3_3()

