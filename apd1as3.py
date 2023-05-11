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
    





