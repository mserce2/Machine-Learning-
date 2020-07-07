# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:09:16 2020

@author: Mete
"""

#%%library
import pandas as pd
import numpy as np

#%%import data
data=pd.read_csv("rfc.csv")
#gereksiz datalardan kurtuluyoruz
data.drop(["id", "Unnamed: 32"],axis=1,inplace=True)

#%%iyi ve kötü huylu kanser anlamına gelen değerleri 
#0ve1 int formatına dönüştürüyoruz 0 iyi 1 kötü
data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%hiyararşik üstünlük olmaması adına değerler arasında;normalizasyon yapıyoruz
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)

#%%
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)
print("soruce:",dt.score(x_test,y_test))
#%%random forest classification
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
print("random forest alfo result:",rf.score(x_test,y_test))




