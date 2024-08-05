#!/usr/bin/env python
# coding: utf-8

# #### Machine Learning Project: Text Source Classification: Try various Neural Network models for training and prediction
# #### 2024 Marc McAllister

# In[ ]:


get_ipython().system('pip install tensorflow pandas scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install nltk')
get_ipython().system('pip install pydot')
#nltk.download('stopwords')


# In[ ]:


import os, pathlib
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from scipy.sparse import csr_matrix

import numpy as np


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import string


# In[ ]:


# Open source path and import text files

folderpath = "sources"
filenames = list()

for name in os.listdir(folderpath):
    if name.endswith('txt'):
        filenames.append(name)

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
    dfsub = dfsub[0:1000]      #Reduce sample size for debugging
    df = pd.concat([df, dfsub])

# Save Data Frame
df.to_csv('Text_Author.csv', index=False)


# In[ ]:


print(titles)


# In[ ]:


# Remove Punctuation
#df['Text'] = df['Text'].str.replace('[^a-zA-Z ]','', regex=1)

# Remove Short Sentences
df = df[df['Text'].str.split().str.len() > 5]

#df.to_csv('Text_Author_Short_Removed.csv', index=False)


# In[ ]:


# Training & Testing Data: Train, Test, Split

train_texts, test_texts, train_labels, test_labels = train_test_split(df['Text'], df['Author'], test_size=0.3, random_state=41)


# In[ ]:


train_texts.shape


# In[ ]:


# Tokenize

#vectorizer = CountVectorizer()
#X_train = vectorizer.fit_transform(train_texts)
#X_test = vectorizer.transform(test_texts)

#vocab_size = len(vectorizer.vocabulary_)


# In[ ]:


# Lets try the Keras Preprocessing Tokenizer instead

tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=5000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
)
tokenizer.fit_on_texts(train_texts)

word_index=tokenizer.word_index

X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=40)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=40)

vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)
print(X_train.shape)


# In[ ]:


# Encode the labels

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_test = label_encoder.transform(test_labels)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


# Let's try some different models:
#   Logistic Regression Classifier      (Reduced Sample) Accuracy: 0.15476984656437626
#   Sequential Model Dense & Dropout    (Reduced Sample) Accuracy: 0.1373
#   Convolutional 1D & Pooling          (Reduced Sample) Accuracy: 0.6205 
#   DNN Sequential Model        (Reduced Sample) Accuracy: 0.6796
#   RNN Sequential Model        (Reduced Sample) Accuracy: 0.4775
#   LSTM Module                 (Reduced Sample) Accuracy: 0.0602


# In[ ]:


# Logistic Regression Classifier

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)


# In[ ]:


# Sequential Model: Dense & Dropout

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
    epochs=10,
    #verbose=False,
    validation_data=(X_test, y_test),
    batch_size=32)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()


# In[ ]:


# Convolutional 1D & Pooling

num_features = vocab_size + 1
embedding_dim = [10]

dropout_rate = 0.1
filters = 512
kernel_size = 10
pool_size = 9

model = models.Sequential()

model.add(layers.Embedding(num_features, 200, input_length=10))
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Conv1D(filters=filters,
    kernel_size=kernel_size,
    activation='relu',
    bias_initializer='random_uniform',
    padding='same'))
model.add(layers.MaxPooling1D(pool_size=pool_size))
model.add(layers.Conv1D(filters=filters * 2,
    kernel_size=kernel_size,
    activation='relu',
    bias_initializer='random_uniform',
    padding='same'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(rate=dropout_rate))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, verbose=True, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()


# In[ ]:


# DNN Sequential Model

model = models.Sequential([
    #layers.Embedding(len(vectorizer.vocabulary_) + 1, 10, input_shape=[10]),
    layers.Embedding(len(tokenizer.word_index) + 1, 200, input_shape=[10]),
    layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train the DNN Sequential Model

history = model.fit(X_train, y_train, epochs=10, verbose=True, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the DNN Sequential model

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()


# In[ ]:


# RNN Sequential Model

model = models.Sequential([
    layers.Embedding(len(tokenizer.word_index) + 1, 10, input_shape=[10]),
    layers.GRU(16),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the RNN Sequential Model

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the RNN Sequential model

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()


# In[ ]:


# LSTM Module

model = keras.models.Sequential()
#model.add(layers.Embedding(len(vectorizer.vocabulary_), 32, input_length=10))
model.add(layers.Embedding(len(tokenizer.word_index), 32, input_length=10))
model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=["accuracy"])
model.summary()

# Train the Model

history = model.fit(X_train, y_train, epochs=10, verbose=True, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

