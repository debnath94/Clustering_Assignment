# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 00:31:18 2022

@author: debna
"""
##Q4

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

tele = pd. read_excel("E:/Clustering/Telco_customer_churn.xlsx")
tele. info()
tele. columns
tele. isna(). sum()
duplicate = tele. duplicated()
sum(duplicate)

#creating instance of labelencoder

labelencoder = LabelEncoder()
x = tele. iloc[:,:]

x['Referred a Friend']= labelencoder. fit_transform(x['Referred a Friend'])
x['Offer']= labelencoder. fit_transform(x['Offer'])
x['Phone Service']= labelencoder. fit_transform(x['Phone Service'])
x['Multiple Lines']= labelencoder. fit_transform(x['Multiple Lines'])
x['Internet Service']= labelencoder. fit_transform(x['Internet Service'])
x['Internet Type']= labelencoder. fit_transform(x['Internet Type'])
x['Online Backup']= labelencoder. fit_transform(x['Online Backup'])
x['Online Security']= labelencoder. fit_transform(x['Online Security'])
x['Device Protection Plan']= labelencoder. fit_transform(x['Device Protection Plan'])
x['Premium Tech Support']= labelencoder. fit_transform(x['Premium Tech Support'])
x['Streaming TV']= labelencoder. fit_transform(x['Streaming TV'])
x['Streaming Movies']= labelencoder. fit_transform(x['Streaming Movies'])
x['Streaming Music']= labelencoder. fit_transform(x['Streaming Music'])
x['Unlimited Data']=labelencoder. fit_transform(x['Unlimited Data'])
x['Contract']= labelencoder. fit_transform(x['Contract'])
x['Paperless Billing']= labelencoder. fit_transform(x['Paperless Billing'])
x['Payment Method'] = labelencoder. fit_transform(x['Payment Method'])


# Univariate and Bivariate analysis on the dataset

plt. hist(tele["Referred a Friend"]) 
plt. hist(tele["Offer"])
plt. hist(tele["Phone Service"])
plt. hist(tele["Monthly Charge"])
plt. hist(tele["Total Charges"])
plt. hist(tele["Total Refunds"])
plt. hist(tele["Total Extra Data Charges"])
plt. hist(tele["Total Revenue"])
plt. hist(tele["Tenure in Months"])
plt. hist(tele["Avg Monthly Long Distance Charges"])
plt. hist(tele["Avg Monthly GB Download"])

##Checking outliers
sns. boxplot(tele["Referred a Friend"])##no outlier

sns. boxplot(tele["Offer"]) ##no outlier
sns. boxplot(tele["Phone Service"]) ## outlier have but it can be ignored
sns. boxplot(tele["Multiple Lines"]) ##no outliers
sns. boxplot(tele["Internet Service"]) ## outlier have but it can be ignored
sns. boxplot(tele["Monthly Charge"]) ##no outlier
sns. boxplot(tele["Total Charges"]) ##no outlier
sns. boxplot(tele["Total Refunds"]) ##outlier have but it can be ignored
sns. boxplot(tele["Total Extra Data Charges"]) ##outlier have but it can be ignored
sns. boxplot(tele["Total Long Distance Charges"]) ##outliers have but it can be ignored
sns. boxplot(tele["Total Revenue"]) ##outliers
sns. boxplot(tele["Tenure in Months"]) ##no outlier
sns. boxplot(tele["Avg Monthly Long Distance Charges"]) ##no outliers
sns. boxplot(tele["Avg Monthly GB Download"]) ##outliers

##outlier treatment by rectify method

from feature_engine. outliers import Winsorizer


winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Total Revenue'])
tele["Total Revenue"] = winsor. fit_transform(tele[["Total Revenue"]])

winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Avg Monthly GB Download'])
tele["Avg Monthly GB Download"] = winsor. fit_transform(tele[["Avg Monthly GB Download"]])


tele. skew(axis = 0, skipna = True) #skewness
tele. kurtosis(axis = 0, skipna = True)    #kurtosis

tele = tele.drop(['Customer ID','Count','Quarter'], axis = 1)
tele. describe()

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

df_norm = norm_func(tele. iloc[:,:])
df_norm. describe()
df_norm.head()


TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ 
mb = pd. Series(model.labels_) 
df_norm['clust'] = mb
df_norm.head()

tele. head()
df_norm.iloc[:, 0:53].groupby(df_norm.clust).mean()

#Cluster1 = these are customers take offer as well. internert used by them is modarate to heavy. and revenue earned
also best.these are the customers that are least likely to churn.

#Cluster2 = these customer dont take offer as well. and revenue earned also not good. these are customer churn most
#cluster3 = These are the customers that stand in the middle of the two extremes and may or may not churn that frequently




