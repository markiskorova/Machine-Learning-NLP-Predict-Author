#!/usr/bin/env python
# coding: utf-8

# #### Machine Learning Project: Train Model to predict the author of a phrase
# #### Marc McAllister
# #### 2024
# 
# ###### Suggestions are welcome. Thank you.

# In[ ]:


get_ipython().system('pip install tensorflow pandas scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install nltk')
nltk.download('stopwords')


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
import pickle

from nltk.tag import pos_tag

import pathlib
import os

from nltk.corpus import stopwords


# In[ ]:


os.getcwd()


# In[ ]:


folderpath = "sources"

filenames = list()

for name in os.listdir(folderpath):
    if name.endswith('txt'):
        filenames.append(name)
        
filenames


# In[ ]:


# Get and Process Data 

titles = []

df = pd.DataFrame(columns=['Author','Text'])

for fname in filenames:
    samplefilepath = 'sources/' + fname
    sampletext = pathlib.Path(samplefilepath).read_text()

    title = sampletext.split('\n')[0]
    authorname = sampletext.split('\n')[1]

    sampletext = sampletext.replace("\n", " ")
    sampletext_sentences = sampletext.split(".")

    titles.append(title)

    i = len(sampletext_sentences)

    data = {'Author': [authorname]*i,
        'Text': sampletext_sentences}

    dfsub = pd.DataFrame(data, columns=['Author','Text'])
    #dfsub = dfsub[0:1000]      #For Small
    df = pd.concat([df, dfsub])


#df.to_csv('Text_Author.csv', index=False)


# In[ ]:


# Start building Model

train_texts, test_texts, train_labels, test_labels = train_test_split(df['Text'], df['Author'], test_size=0.2, random_state=42)


# In[ ]:


print(train_texts.size)
print(test_texts.size)
print(train_labels.size)
print(test_labels.size)


# In[ ]:


train_texts[0:10]


# In[ ]:


train_labels[0:10]


# In[ ]:


# Tokenize and vectorize text data

#vectorizer = CountVectorizer(ngram_range=(2,3))

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_texts)
x_test = vectorizer.transform(test_texts)


x_train


# In[ ]:


len(vectorizer.vocabulary_)


# In[ ]:


# Encode the labels

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_test = label_encoder.transform(test_labels)

#print(label_encoder)
y_train


# In[ ]:


# Let's Build the model

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)


# In[ ]:


# Convert sparse matrices

x_train_sparse = tf.convert_to_tensor(csr_matrix(x_train).todense(), dtype=tf.float32)
x_val_sparse = tf.convert_to_tensor(csr_matrix(x_val).todense(), dtype=tf.float32)
x_test_sparse = tf.convert_to_tensor(csr_matrix(x_test).todense(), dtype=tf.float32)


# In[ ]:


print(x_train_sparse.shape)
print(x_val_sparse.shape)
print(x_test_sparse.shape)


# In[ ]:


# Train the Model

history = model.fit(x_train_sparse, y_train, epochs=10, batch_size=32, validation_data=(x_val_sparse, y_val))


# In[ ]:


print(history)


# In[ ]:


# Evaluate the model

predictions = model.predict(x_test)
y_pred = predictions.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# In[ ]:


#model.save('tensorflow_detection_model.keras')

keras.saving.save_model(model, 'tensorflow_detection_model.keras', overwrite=True)


# In[ ]:


label_encoder.classes_


# In[ ]:


test_text = 'into the past'

print(test_text)

tvector = vectorizer.transform([test_text])
tsparse = tf.convert_to_tensor(tvector.todense(), dtype=tf.float32)
pred = model.predict(tsparse)

print(pred)
print(np.argmax(pred))

pred_label_index = np.argmax(pred)
pred_label = label_encoder.classes_[pred_label_index]

print(pred_label)


# Suggestions on accuracy?
