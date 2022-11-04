# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:47:52 2022

@author: debna
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

data = pd.read_excel("E:/Clustering/EastWestAirlines.xlsx",sheet_name= 1)

data.shape
data.info()
data.describe()
data. columns
##data Cleansing###
data = data. rename(columns ={'ID#': 'ID', 'Award?': 'Award'})
data. isna(). sum()
data. var()
data. var()==0
duplicate = data.duplicated()
duplicate
sum(duplicate)

data = data. drop(["ID"], axis=1)
sns. boxplot(data.Balance)

IQR = data['Balance'].quantile(0.75) - data['Balance'].quantile(0.25)
IQR
lower_limit = data['Balance'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['Balance'].quantile(0.75) + (IQR * 1.5)
upper_limit

from feature_engine. outliers import Winsorizer


winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Balance'])
data_t= winsor. fit_transform(data[['Balance']])
sns. boxplot(data_t.Balance)
winsor.left_tail_caps_, winsor.right_tail_caps_

sns. boxplot(data.Qual_miles)

IQR = data['Qual_miles'].quantile(0.75) - data['Qual_miles'].quantile(0.25)
IQR
lower_limit = data['Qual_miles'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['Qual_miles'].quantile(0.75) + (IQR * 1.5)
upper_limit

winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Qual_miles'])
data_Q= winsor. fit_transform(data[['Qual_miles']])
sns. boxplot(data_Q.Qual_miles)

data.describe()
def norm_func(i):
    x= (i- i.min())/ (i.max()-i.min())
    return(x)

data_norm = norm_func(data. iloc[:, 1:])
data_norm. describe()

from scipy. cluster. hierarchy import linkage, dendrogram

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(20, 9));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10
    )

from sklearn. cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=5, linkage = 'complete', affinity = "euclidean"). fit(data_norm)

h_complete. labels_

cluster_labels = pd. Series(h_complete.labels_)
data.head()
data['clust'] = cluster_labels
data = data.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10,]]
data.head()
data. iloc[:,0:]. groupby(data.clust). mean()
data. iloc[:,0:]. groupby(data.clust). std()









