# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:43:48 2022

@author: debna
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:32:23 2022

@author: debna
"""
import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv("E:/Clustering/New folder/crime_data.csv")
##Data Cleansing###
data.shape
data.info()
data. describe()

data. Murder = data. Murder. astype('int64')
data. Rape = data. Rape. astype('int64')
###check for null values
data. isna(). sum()
##chek variance
data. var()
data.var() ==0
##checking duplicate values

duplicate = data.duplicated()
duplicate
sum(duplicate)
###Check for outliers and shape of data distribution

import numpy as np
plt. bar(height = data, x= np.arange(1,774,1))
plt.hist(data.Murder)
import seaborn as sns
sns.boxplot(data.Murder)
sns.boxplot(data.Assault)
sns.boxplot(data.UrbanPop)
sns.boxplot(data.Rape)

IQR= data['Rape'].quantile(0.75)- data['Rape'].quantile(0.25)
lower_limit= data['Rape'].quantile(0.25)-(IQR*1.5)
upper_limit= data['Rape'].quantile(0.75)+(IQR*1.5)

outliers_data= np.where(data['Rape']>upper_limit, True, np.where(data['Rape']<lower_limit, True,False))
data_trimmed= data.loc[~(outliers_data),]
data.shape,data_trimmed.shape
sns.boxplot(data_trimmed.Rape)








####Clastering##

data.describe()

def norm_func(i):
    x= (i- i.min()) / (i.max()-i.min())
    return(x)

data_norm = norm_func(data. iloc[:, 2:])
data_norm. describe()

from scipy. cluster. hierarchy import linkage, dendrogram

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10
    )

plt.show()

from sklearn. cluster import AgglomerativeClustering
from sklearn. cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage = 'complete', affinity = "euclidean"). fit(data_norm)

h_complete. labels_

cluster_labels = pd. Series(h_complete.labels_)

data['clust'] = cluster_labels
data = data.iloc[:,[0,5,2,3,4,1]]
data.head()
data. iloc[:,2:]. groupby(data.clust). mean()
data. iloc[:,2:]. groupby(data.clust). std()






