# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 22:53:51 2022

@author: debna
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
data= pd.read_csv("E:/Clustering/AutoInsurance.csv")
data.shape
data.dtypes
data.info()
data = data. rename(columns ={'Total Claim Amount': 'Total_Claim_Amount', 'Customer Lifetime Value': 'Customer_Lifetime_Value',
                              'Monthly Premium Auto':'Monthly_Premium_Auto','Months Since Last Claim':'Months_Since_Last_Claim','Months Since Policy Inception':'Months_Since_Policy_Inception','Number of Open Complaints':'Number_of_Open_Complaints','Number of Policies':'Number_of_Policies'})

data. columns
data.Customer_Lifetime_Value= data.Customer_Lifetime_Value.astype('int64')
data.Total_Claim_Amount= data.Total_Claim_Amount. astype('int64')
duplicate= data. duplicated()
duplicate
sum(duplicate)
data. var()
data.var()==0
data.var(axis=0)== 0
data.isna(). sum()



data = data[['Customer','Sales Channel', 'Vehicle Class', 'Vehicle Size','State', 'Response', 'Coverage', 'Education' ,
            'EmploymentStatus', 'Gender', 'Location Code', 'Marital Status', 'Policy Type',
            'Policy','Renew Offer Type', 'Customer_Lifetime_Value',
            'Income','Monthly_Premium_Auto',
            'Months_Since_Last_Claim', 'Months_Since_Policy_Inception',
            'Number_of_Open_Complaints', 'Number_of_Policies','Total_Claim_Amount']]

from sklearn. preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X = data.iloc[:,:]

X['State'] = labelencoder. fit_transform(X['State'])
X['Response'] = labelencoder. fit_transform(X['Response'])
X['Coverage'] = labelencoder. fit_transform(X['Coverage'])
X['Location Code'] = labelencoder. fit_transform(X['Location Code'])
X['Coverage'] = labelencoder. fit_transform(X['Coverage'])
X['Education'] = labelencoder. fit_transform(X['Education'])
X['EmploymentStatus'] = labelencoder. fit_transform(X['EmploymentStatus'])
X['Gender'] = labelencoder. fit_transform(X['Gender'])
X['Location Code'] = labelencoder. fit_transform(X['Location Code'])
X['Marital Status'] = labelencoder. fit_transform(X['Marital Status'])
X['Policy Type'] = labelencoder. fit_transform(X['Policy Type'])
X['Renew Offer Type'] = labelencoder. fit_transform(X['Renew Offer Type'])
X['Policy'] = labelencoder. fit_transform(X['Policy'])
X['Renew Offer Type'] = labelencoder. fit_transform(X['Renew Offer Type'])

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_norm = norm_func(data .iloc[:, 16:])
df_norm.describe()
from scipy.cluster.hierarchy import linkage, dendrogram
z = linkage(df_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10
)
plt.show()


from sklearn. cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters= 4, linkage = 'complete', affinity = 'euclidean'). fit(df_norm)
h_complete.labels_

cluster_labels = pd. Series(h_complete.labels_)



data['clust'] = cluster_labels
data = data. iloc[:, [23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
data.head()

data. iloc[:,17:]. groupby(data. clust). mean()



































