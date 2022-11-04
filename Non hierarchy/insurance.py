# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:47:44 2022

@author: debna
"""
###Qno=5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

insurance = pd. read_csv("E:/Clustering/AutoInsurance.csv")
insurance. columns
insurance. isna(). sum()
insurance. dtypes
duplicate = insurance. duplicated()
sum(duplicate)

from sklearn. preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x = insurance. iloc[:,:]
x['State'] = labelencoder. fit_transform(x['State'])
x['Response'] = labelencoder. fit_transform(x['Response'])
x['Coverage'] = labelencoder. fit_transform(x['Coverage'])
x['Education'] = labelencoder. fit_transform(x['Education'])
x['EmploymentStatus'] = labelencoder. fit_transform(x['EmploymentStatus'])
x['Gender'] = labelencoder. fit_transform(x['Gender'])
x['Location Code'] = labelencoder.fit_transform(x['Location Code'])
x['Marital Status'] = labelencoder. fit_transform(x['Marital Status'])
x['Policy Type'] = labelencoder. fit_transform(x['Policy Type'])
x['Policy'] = labelencoder. fit_transform(x['Policy'])
x['Renew Offer Type'] = labelencoder. fit_transform(x['Renew Offer Type'])
x['Sales Channel'] = labelencoder. fit_transform(x['Sales Channel'])
x['Vehicle Class'] = labelencoder. fit_transform(x['Vehicle Class'])
x['Vehicle Size'] = labelencoder. fit_transform(x['Vehicle Size'])

insurance = insurance. iloc[:,[0,6,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

insurance. info()

##outlier analysis##

sns.boxplot(insurance['Customer Lifetime Value'])
from feature_engine. outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Customer Lifetime Value'])

insurance['Customer Lifetime Value']=winsor. fit_transform(insurance[['Customer Lifetime Value']])
sns.boxplot(insurance['Customer Lifetime Value'])

sns. boxplot(insurance['Income'])##no outlier

sns. boxplot(insurance['Monthly Premium Auto'])

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Monthly Premium Auto'])
insurance['Monthly Premium Auto']=winsor. fit_transform(insurance[['Monthly Premium Auto']])

sns. boxplot(insurance['Months Since Last Claim'])##no outlier
sns. boxplot(insurance['Months Since Policy Inception'])

sns. boxplot(insurance['Total Claim Amount'])
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Total Claim Amount'])
insurance['Total Claim Amount']=winsor. fit_transform(insurance[['Total Claim Amount']])

insurance. var()
insurance. var() == 0 ##no variance
insurance. describe()

def norm_func(i):
    x = (i - i.min())/ (i. max() - i. min())
    return(x)

df_norm = norm_func(insurance. iloc[:,2:])
df_norm. describe()

TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans. fit(df_norm)
    TWSS. append(kmeans. inertia_)
    
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model = KMeans(n_clusters = 4)
model.fit(df_norm)
model.labels_
mb = pd.Series(model.labels_)
insurance['clust'] = mb
insurance. head()
df_norm.head()

insurance=insurance. drop(["Effective To Date"], axis=1)
insurance = insurance.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
insurance.head()

insurance. iloc [:,2:]. groupby(insurance. clust). mean()

insurance. to_csv("auto_insurance.csv", encoding = "utf-8")
import os
os. getcwd()

#cluster0 = people those who are low income prone to be fraud transcations, they might be the customers  that cause huge loss of company

#cluster1 = same as cluster 0

#cluster 2 = those people might or might not give a fraud claim. but take more observations
#cluster 3 = those people have high income and less prone to be a fraud transcations


