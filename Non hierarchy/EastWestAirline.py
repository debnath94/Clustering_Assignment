# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:33:40 2022

@author: debna
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

data = pd. read_excel("E:/Clustering/EastWestAirlines.xlsx", sheet_name = 1)

data.shape
data.info()
data.describe()
data = data. rename(columns ={'ID#': 'ID', 'Award?': 'Award'})
data. columns

data. isna(). sum()

duplicate = data.duplicated()
duplicate
sum(duplicate)

##outlier treatment
sns.boxplot(data.Balance)

IQR = data['Balance']. quantile(0.75)- data['Balance']. quantile(0.25)
lower_limit = data['Balance']. quantile(0.25)-(IQR*1.5)
upper_limit = data['Balance']. quantile(0.75) + (IQR*1.5)

from feature_engine. outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Balance'])
data['Balance']=winsor. fit_transform(data[['Balance']])

sns. boxplot(data. Qual_miles)
IQR = data['Qual_miles']. quantile(0.75)- data['Qual_miles']. quantile(0.25)
lower_limit = data['Qual_miles']. quantile(0.25) - (IQR*1.5)
upper_limit = data['Qual_miles']. quantile(0.75) + (IQR*1.5)

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Qual_miles'])

data['Qual_miles']=winsor. fit_transform(data[['Qual_miles']])



sns. boxplot(data. cc1_miles)##no outlier
sns. boxplot(data. cc2_miles)
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['cc2_miles'])

data['cc2_miles']=winsor. fit_transform(data[['cc2_miles']])

sns. boxplot(data. cc3_miles)

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['cc3_miles'])

data['cc3_miles']=winsor. fit_transform(data[['cc3_miles']])

sns. boxplot(data. Bonus_miles)
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Bonus_miles'])

data['Bonus_miles']=winsor. fit_transform(data[['Bonus_miles']])


sns. boxplot(data.Flight_miles_12mo)
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Flight_miles_12mo'])

data['Flight_miles_12mo']=winsor. fit_transform(data[['Flight_miles_12mo']])

sns. boxplot(data.Bonus_trans)
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Bonus_trans'])

data['Bonus_trans']=winsor. fit_transform(data[['Bonus_trans']])

sns. boxplot(data. Flight_trans_12)
winsor=Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Flight_trans_12'])

data['Flight_trans_12']=winsor. fit_transform(data[['Flight_trans_12']])
  
sns. boxplot(data.Days_since_enroll)##no outlier
sns. boxplot(data.Flight_trans_12)
winsor=Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Flight_trans_12'])

data['Flight_trans_12']=winsor. fit_transform(data[['Flight_trans_12']]) 

data. var()
data. var()==0

data = data. drop(["Qual_miles", "cc2_miles", "cc3_miles", "Flight_miles_12mo", "Flight_trans_12"], axis = 1)

data. describe()

def norm_func(i):
    x = (i - i.min())/ (i. max()-i.min())
    return(x)
df_norm = norm_func(data. iloc[:,1:])
df_norm. describe()

###Univariate analysis
plt. hist(data. Balance)
plt.hist(data.Days_since_enroll)
plt. hist(data. Bonus_miles)

#Bivariate analysis
plt.scatter(data["Balance"], data["Days_since_enroll"])
plt.xlabel('Days_since_enroll')
plt.ylabel('Balance')
plt. show

data.skew(axis = 0, skipna = True) 
data.kurtosis(axis = 0, skipna = True)

from sklearn. cluster import KMeans
TWSS = []
k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans. fit(df_norm)
    TWSS. append(kmeans. inertia_)
    
TWSS
##scree plot

plt. plot(k, TWSS,'ro-'); plt. xlabel("no_of_clusters"); plt. ylabel("total_within_ss")

model = KMeans(n_clusters = 4)
model.fit(df_norm)
model.labels_
mb = pd. Series(model. labels_)
data['clust'] = mb
data. head()

data = data. iloc[:,[7,0,1,2,3,4,5,6,]]













