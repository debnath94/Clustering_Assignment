# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 00:35:14 2022

@author: debna
"""

import pandas as pd
import matplotlib.pylab as plt
data = pd.read_excel("E:/Clustering/Telco_customer_churn.xlsx")
data.shape
data.dtypes
data.info()
data. isna(). sum()
duplicate= data.duplicated()
duplicate
sum(duplicate)
data.describe()


data= data. drop(["Count","Quarter","Referred a Friend","Number of Referrals","Tenure in Months","Contract","Paperless Billing","Payment Method"], axis=1)
data.columns
data= data[['Customer ID', 'Offer', 'Phone Service','Unlimited Data', 'Multiple Lines','Internet Service', 'Internet Type', 'Streaming Music','Online Security', 'Online Backup', 'Device Protection Plan','Premium Tech Support', 'Streaming TV', 'Streaming Movies','Avg Monthly GB Download', 'Avg Monthly Long Distance Charges', 'Monthly Charge', 'Total Charges','Total Refunds', 'Total Extra Data Charges','Total Long Distance Charges', 'Total Revenue']]

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

data_norm= norm_func(data. iloc[:, 15:])
data_norm.describe()
from scipy.cluster.hierarchy import linkage, dendrogram
z= linkage(data_norm, method= "complete", metric= "euclidean")
plt.figure(figsize= (15,8)); plt.title('Hierarchical clustering Dendrogram'); plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z,
           leaf_rotation = 0,
           leaf_font_size = 10
)           
          
from sklearn. cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters= 4, linkage= 'complete', affinity= "euclidean").fit(data_norm)
h_complete.labels_
cluster_labels= pd. Series(h_complete.labels_)
data['clust']= cluster_labels
data= data.iloc[:,[22,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,]]
data.head()
data.iloc[:, 16:]. groupby(data.clust). mean()
data.iloc[:, 16]. groupby(data.clust).std()

data.to_csv("CHURN.csv", encoding= "utf-8")
import os
os.getcwd()









