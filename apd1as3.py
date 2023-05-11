# -*- coding: utf-8 -*-
"""
Created on Sun May  7 03:51:45 2023

@author: manus
"""

# importing the required libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt


def read_file(file_name, col, indicator, yrs):
    '''
    defining a function that returns two dataframes,
    one with countries as columns and other with years
    as columns. Function takes file name and the required
    filters as its attribute
    '''
    # reading the file
    df0 = pd.read_csv(file_name, skiprows=4)
    # cleaning the dataframe
    df1 = df0.groupby(col, group_keys=True)
    df1 = df1.get_group(indicator)
    df1 = df1.reset_index()
    a = df1['Country Name']
    df1 = df1.iloc[25:125, yrs]
    df1 = df1.dropna(axis=0)
    df1.insert(loc=0, column='Country Name', value=a)
    # taking the transpose
    df2 = df1.set_index('Country Name').T
    return df1, df2


# mentioning the list of required years
years = [35, 36, 37, 38, 39, 40]

# creating the required dataframe for clustering
co2_1, co2_2 = read_file("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv",
                         "Indicator Name", "CO2 emissions (metric tons per capita)", years)

# printing the created dataframes
print(co2_1)
print(co2_2)

# returns a numpy array x
x = co2_1.iloc[:, 1:].values


def normalising(value):
    '''
    defining a function that 
    uses the MinMaxScaler to
    normalise the required data
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    # fitting the array for normalising
    x_scaled = min_max_scaler.fit_transform(value)
    # creating a dataframe holding the normalised values
    df = pd.DataFrame(x_scaled)
    return df


# creating the required normalised data
n_df = normalising(x)
print(n_df)


def n_cluster(dataframe, n):
    '''
    defining a function to find the number 
    of clusters required using the elbow method
    '''
    # getting the within-cluster sum of squares
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(dataframe)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(n_df, 10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), k)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# finding k-means cluster
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100,
                n_init=10, random_state=0)

# fitting and prediction using k-mean clustering
labels = kmeans.fit_predict(n_df)

# finding centroids for Kmean cluster
cen = kmeans.cluster_centers_
print('centroids=', cen)


'''
Ploting Kmeans clusters
after finding the required number of
clusters as 3
'''
plt.figure(figsize=(10, 5))
# Ploting cluster 1
plt.scatter(n_df.values[labels == 0, 0], n_df.values[labels ==
            0, 1], s=100, c='green', label='Cluster1')
# Ploting cluster 2
plt.scatter(n_df.values[labels == 1, 0], n_df.values[labels ==
            1, 1], s=100, c='orange', label='Cluster2')
# Ploting cluster 3
plt.scatter(n_df.values[labels == 2, 0], n_df.values[labels ==
            2, 1], s=100, c='red', label='Cluster3')
# Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=100, c='blue', label='Centroids')
# setting up the legends,title and the labels
plt.legend()
plt.title('Clusters of CO2 emission in metric tons per capita of 100 Countries')
plt.xlabel('Countries')
plt.ylabel('Concentration')
plt.show()

# adding the cluster label column to the dataframe
co2_1['labels'] = labels

# making a list of years required for curve fitting
yers = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
# reading in the data for curve fitting
gdp_1, gdp_2 = read_file("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv",
                         "Indicator Name", "GDP growth (annual %)", yers)
# adding the means column to the dataframe
gdp_2['mean'] = gdp_2.mean(axis=1)
# adding the years column from the index
gdp_2['years'] = gdp_2.index

print(gdp_2)

# Ploting the data of GDP growth (annual %)
ax = gdp_2.plot(x='years', y='mean', figsize=(
    10, 5), title='Mean gdp of 100 country ', xlabel='Years', ylabel='Mean')


def exponential(t, n0, g):
    """defining function to calculate exponential function with scale factor n0 and growth rate g"""
    t = t - 1999.0
    ef = n0 * np.exp(g*t)
    return ef


# converting string type years to numeric type
gdp_2["years"] = pd.to_numeric(gdp_2["years"])

# fitting exponential fit
param, covar = opt.curve_fit(
    exponential, gdp_2["years"], gdp_2["mean"], p0=(1993.0, 1.9149))

gdp_2["fit"] = exponential(gdp_2["years"], *param)

gdp_2.plot("years", ["mean", "fit"],
           title='Data fitting using exponential function',
           figsize=(10, 5))
plt.show()
print(gdp_2)


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


# fitting logistic fit
param, covar = opt.curve_fit(logistic, gdp_2["years"], gdp_2["mean"],
                             p0=(3e12, 1.9149, 1993.0), maxfev=5000)

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
gdp_2["logistic function fit"] = logistic(gdp_2["years"], *param)
gdp_2.plot("years", ["mean", "fit"],
           title='Data fitting using logistic function',
           figsize=(10, 5))
plt.show()

# predicting GDP/year over the years till 2020
year = np.arange(1990, 2020)
print(year)
forecast = logistic(year, *param)
print('forecast=', forecast)


# using plt function to plot the required plots
plt.figure()
plt.plot(gdp_2["years"], gdp_2["mean"], label="GDP")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP/year")
plt.legend()
plt.title('Prediction of GDP from 1990 to 2020')
plt.show()

# using the given err_ranges function


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


low, up = err_ranges(year, logistic, param, sigma)

# plotting using the plt function
plt.figure()
plt.plot(gdp_2["years"], gdp_2["mean"], label="GDP")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="cyan", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()
