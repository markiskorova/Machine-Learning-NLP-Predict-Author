#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow pandas scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install nltk')
get_ipython().system('pip install pydot')
#nltk.download('stopwords')


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras import layers, models

import os, pathlib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords



# In[ ]:


# Hyperparameters

vocab_size = 5000
embedding_dim = 128
max_length = 40
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

STOPWORDS = set(stopwords.words('english'))


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
    #dfsub = dfsub[0:1000]      #Reduce sample size for debugging
    df = pd.concat([df, dfsub])

# Remove Short Sentences
df = df[df['Text'].str.split().str.len() > 5]

# Save Data Frame
df.to_csv('Text_Author.csv', index=False)


# In[ ]:


print(df.shape)


# In[ ]:


articles = df['Text'].to_list()
labels = df['Author'].to_list()


# In[ ]:


# Or open saved dataframe (freeze this cell otherwise)

articles = []
labels = []

with open('Text_Author.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        articles.append(article)
print(len(labels))
print(len(articles))


# In[ ]:


train_articles, validation_articles, train_labels, validation_labels = train_test_split(articles, labels, test_size=1-training_portion, random_state=42)


# In[ ]:


print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))


# In[ ]:


# CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,3))
#vectorizer.max_features = vocab_size

vectorizer.fit(train_articles)

#vocab_size = len(vectorizer.vocabulary_)

#print(vectorizer.vocabulary_)
print(len(vectorizer.vocabulary_))

train_sequences = vectorizer.transform(train_articles)
validation_sequences = vectorizer.transform(validation_articles)

#print(train_sequences.shape)
print(train_articles[0])
print(train_sequences[0])


# In[ ]:


# Keras Tokenizer

tokenizer = Tokenizer(
    num_words=vocab_size,
    filters='“”‘’!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    oov_token=oov_tok,
)

#tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

#dict(list(word_index.items())[0:10])
print(len(tokenizer.word_index))

train_sequences = tokenizer.texts_to_sequences(train_articles)
validation_sequences = tokenizer.texts_to_sequences(validation_articles)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#print(train_sequences.shape)
print(train_articles[0])
print(train_sequences[0])


# In[ ]:


set(labels)


# In[ ]:


label_encoder = LabelEncoder()
label_encoder.fit(labels)
training_label_enc = label_encoder.transform(train_labels)
validation_label_enc = label_encoder.transform(validation_labels)


# In[ ]:


X_train = train_padded
y_train = training_label_enc
X_test = validation_padded
y_test = validation_label_enc


# In[ ]:


print(X_train[1])


# In[ ]:


model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

layers.GRU(32),
layers.Dense(len(label_encoder.classes_), activation='softmax')


# In[ ]:


len(tokenizer.word_index)


# In[ ]:


# Embedding

model = models.Sequential([
    layers.Embedding(len(tokenizer.word_index) + 1, 64, input_shape=[len(tokenizer.word_index), ]),
    #layers.Bidirectional(layers.LSTM(64)),
    layers.Bidirectional(layers.GRU(256)),
    layers.Dense(len(label_encoder.classes_), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# In[ ]:


callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.1, patience=5)

# Train the Model

history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    verbose=True, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    #callbacks=[callback]
)

# Evaluate the model

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

