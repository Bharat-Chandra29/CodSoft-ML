#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from tqdm import tqdm


# In[3]:


data=pd.read_csv('spam.csv',encoding='latin-1')


# In[4]:


data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})  # This step could be simplified
X = data['v2']
Y = data['label']


# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[6]:


print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


tfidf_vectorizer = TfidfVectorizer()


# In[9]:


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)


# In[10]:


classifier=MultinomialNB()


# In[11]:


classifier.fit(X_train_tfidf,Y_train)


# In[12]:


X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[13]:


Y_pred = classifier.predict(X_test_tfidf)


# In[14]:


accuracy = accuracy_score(Y_test, Y_pred)


# In[15]:


report = classification_report(Y_test, Y_pred, target_names=['Legitimate SMS', 'spam SMS'])


# In[16]:


progress_bar = tqdm(total=100, position=0, leave=True)


# In[17]:


for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')


# In[18]:


progress_bar.close()


# In[19]:


print(f'Accuracy: {accuracy:.2f}')


# In[20]:


print('Classification Report:')
print(report)


# In[ ]:




