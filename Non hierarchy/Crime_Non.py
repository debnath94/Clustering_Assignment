# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:33:36 2022

@author: debna
"""

import pandas as pd
import matplotlib. pylab as plt

crime = pd. read_csv("E:/Clustering/New folder/crime_data.csv")
crime. head()
crime.columns= ["State","Murder","Assault","Urbanpop","Rape"]
crime. head()

crime. isna().sum()
crime1 = crime.duplicated()
sum(crime1)

##Univibrate Analysis
import matplotlib. pyplot as plt

plt.hist(crime.Murder)
plt.hist(crime.Assault)
plt.hist(crime.Urbanpop)
plt.hist(crime.Rape)
from scipy import stats
crime. Murder. skew()
crime. Rape. skew()

##Outlier treatment##
import seaborn as sns
import numpy as np

sns. boxplot(crime. Murder) #no outlier
sns. boxplot(crime. Assault)#no outlier
sns. boxplot(crime. Urbanpop)#no outlier
sns. boxplot(crime. Rape) ##outlier

IQR = crime['Rape'].quantile(0.75) - crime['Rape'].quantile(0.25)
IQR
lower_limit = crime['Rape'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = crime['Rape'].quantile(0.75) + (IQR * 1.5)
upper_limit
crime['Rape'] = np.where(crime['Rape'] >= upper_limit, upper_limit, crime['Rape'])
sns.boxplot(crime.Rape)

crime. var()== 0 ##Checking for zero variance values
crime. describe()

def norm_func(i):
    x= (i-i. min())/(i. max()-i.min())
    return(x)

def_norm = norm_func(crime. iloc[:,1:])

def_norm. describe()

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(def_norm)
    TWSS.append(kmeans.inertia_)

TWSS
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
model = KMeans(n_clusters = 3)
model.fit(def_norm)
model.labels_
mb = pd.Series(model.labels_) 
crime['clust'] = mb
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

crime.iloc[:, 1:6].groupby(crime.clust).mean()

#Cust0 = crime rate is very high especially Assault. we need to provide more security for public places
#clust1 = crime rate is less the clust0 but need to provide security for public safety
#clust1 = crime rate is very low. so provide less security

