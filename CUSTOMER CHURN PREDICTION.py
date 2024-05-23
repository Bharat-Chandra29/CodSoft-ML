#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,r2_score,precision_score,recall_score,f1_score


# In[3]:


df=pd.read_csv('Churn_Modelling.csv',header=0)


# In[4]:


df.info()


# In[5]:


df.isnull().values.any()


# In[6]:


print(df["Geography"].unique())
print(df["Gender"].unique())
print(df["NumOfProducts"].unique())
print(df["HasCrCard"].unique())
print(df["IsActiveMember"].unique())
print(df["Exited"].unique())


# In[7]:


df.head()


# In[8]:


df.drop(labels=["RowNumber","CustomerId","Surname"],axis=1,inplace=True)


# In[9]:


df = pd.get_dummies(df, drop_first=True)


# In[10]:


df.head()


# In[11]:


X=df.drop("Exited",axis=1)
y=df["Exited"]


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[13]:


Scaler=StandardScaler()
X_train=Scaler.fit_transform(X_train)
X_test=Scaler.transform(X_test)


# In[14]:


lr_model=LogisticRegression()


# In[16]:


lr_model.fit(X_train,y_train)


# In[17]:


lr_predictions=lr_model.predict(X_test)


# In[18]:


print("Logistic Regression Model:")
print(confusion_matrix(y_test,lr_predictions))
print(classification_report(y_test,lr_predictions))
print("Accuracy:",accuracy_score(y_test,lr_predictions))
print("r2_score:",r2_score(y_test,lr_predictions))
print("precision_score:",precision_score(y_test,lr_predictions))
print("Recall_score:",recall_score(y_test,lr_predictions))
print("f1_score:",f1_score(y_test,lr_predictions))


# In[ ]:




