#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# In[25]:


genre_list=['action','adult','animation','biography','comedy','crime','documnetry','family','fantasy','game-show','history','horror','music','musical','mystery','news','reality-tv','romance','sci-fi','short','sport','talk-show','thriller','war','western']


# In[26]:


fallback_genre='Unknown'


# In[27]:


try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('train_data.txt.zip', sep='111', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise
print("Shape of train_data:", train_data.shape)
print(train_data.head())
print(train_data.isnull().sum())


# In[28]:


X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
print("Shape of X_train:", X_train.shape)
train_data['GENRE'] = train_data['GENRE'].fillna('')
print(train_data.head())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
print("Number of genre labels:", len(genre_labels))
print(genre_labels[:5])
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(genre_labels)
assert len(X_train) == len(genre_labels), "Number of samples in X_train and genre_labels do not match."


# In[29]:


tfidf_vectorizer = TfidfVectorizer(max_features=5000)


# In[30]:


with tqdm(total=50, desc="Vectorizing Training Data") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)
print("Shape of X_train_tfidf:", X_train_tfidf.shape)


# In[31]:


with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier(naive_bayes)
    multi_output_classifier.fit(X_train_tfidf, Y_train)
    pbar.update(50)


# In[32]:


try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('test_data.txt.zip', sep='111', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")
    raise


# In[33]:


X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())


# In[34]:


with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    pbar.update(50)
    print("Shape of X_test_tfidf:", X_test_tfidf.shape)


# In[35]:


with tqdm(total=50, desc="Predicting Test Data") as pbar:
    Y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)


# In[36]:


test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(Y_pred)
test_results = pd.DataFrame({
    'MOVIE_NAME': test_movie_names,
    'PREDICTED_GENRES': ['; '.join(genres) for genres in predicted_genres]
})


# In[37]:


fallback_genre = 'Unknown' 
test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)


# In[38]:


import zipfile


# In[39]:


output_filename = "test_data_solution.txt"


# In[40]:


with zipfile.ZipFile("test_data_solution.txt.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
    with zip_file.open("test_data_solution.txt", "w") as output_file:
        for index, row in test_results.iterrows():
            movie_name = row['MOVIE_NAME']
            genre_str = ', '.join(row['PREDICTED_GENRES'])
            output_file.write(f"{movie_name} ::: {genre_str}\n".encode('utf-8'))


# In[41]:


Y_train_pred = multi_output_classifier.predict(X_train_tfidf)


# In[42]:


accuracy=accuracy_score(Y_train,Y_train_pred)
precision=precision_score(Y_train,Y_train_pred,average='micro')
recall=recall_score(Y_train,Y_train_pred,average='micro')
f1=f1_score(Y_train,Y_train_pred,average='micro')


# In[43]:


with open("test_data_solution.txt", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nEvaluation:\n")
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1 Score: {f1:.2f}\n")


# In[44]:


print("Evaluation has been saved to 'test_data_solution.txt'.")


# In[45]:


with open('test_data_solution.txt', 'r', encoding='utf-8') as file:
    saved_data = file.read()
print(saved_data)


# In[ ]:





# In[ ]:




