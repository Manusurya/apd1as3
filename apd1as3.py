# -*- coding: utf-8 -*-
"""
Created on Sun May  7 03:51:45 2023

@author: manus
"""

#importing the required libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt


def read_file(file_name, col, indicator,yrs):
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
    df1 = df1.iloc[25:125,yrs]
    df1 = df1.dropna(axis=0)
    df1.insert(loc=0, column='Country Name', value=a)
    # taking the transpose
    df2 = df1.set_index('Country Name').T
    return df1, df2
    

# mentioning the list of required years
years= [35,36,37,38,39,40]

#creating the required dataframe for clustering
co2_1, co2_2 = read_file("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv",
                                       "Indicator Name", "CO2 emissions (metric tons per capita)",years)

#printing the created dataframes
print(co2_1)
print(co2_2)

#returns a numpy array x
x = co2_1.iloc[:,1:].values


def normalising(value):
    '''
    defining a function that 
    uses the MinMaxScaler to
    normalise the required data
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitting the array for normalising
    x_scaled = min_max_scaler.fit_transform(value)
    #creating a dataframe holding the normalised values
    df = pd.DataFrame(x_scaled)
    return df


#creating the required normalised data
n_df = normalising(x)
print(n_df)


def n_cluster(dataframe,n):
    '''
    defining a function to find the number 
    of clusters required using the elbow method
    '''
    #getting the within-cluster sum of squares
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataframe)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(n_df,10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(10,5))
plt.plot(range(1, 10), k)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#finding k-means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and prediction using k-mean clustering
labels = kmeans.fit_predict(n_df)

#finding centroids for Kmean cluster
cen= kmeans.cluster_centers_
print('centroids=',cen)


'''
Ploting Kmeans clusters
after finding the required number of
clusters as 3
'''
plt.figure(figsize=(10,5))
#Ploting cluster 1
plt.scatter(n_df.values[labels == 0, 0], n_df.values[labels == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(n_df.values[labels == 1, 0], n_df.values[labels == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(n_df.values[labels == 2, 0], n_df.values[labels == 2, 1], s = 100, c = 'red', label = 'Cluster3')
#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='blue', label = 'Centroids')
#setting up the legends,title and the labels
plt.legend()
plt.title('Clusters of CO2 emission in metric tons per capita of 100 Countries')
plt.xlabel('Countries')
plt.ylabel('Concentration')
plt.show()





