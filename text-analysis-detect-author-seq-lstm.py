#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow pandas scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install nltk')
get_ipython().system('pip install pydot')
get_ipython().system('pip install keras-self-attention')
get_ipython().system('pip install keras.backend')

#nltk.download('stopwords')


# In[1]:


#mport tensorflow as tf

from tensorflow import keras
from keras import layers, models

#from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

import os, pathlib
import pandas as pd

#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from nltk.corpus import stopwords

#import tensorflow.keras.backend as K


# In[2]:


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
df = df[df['Text'].str.split().str.len() > 4]

# Shuffle data
df = df.sample(frac=1)

# Save Data Frame
df.to_csv('Text_Author.csv', index=False)

articles = df['Text'].to_list()
labels = df['Author'].to_list()


# In[3]:


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


# In[4]:


# Hyperparameters

vocab_size = 20000
embedding_dim = 256
max_length = 32

#oov_tok = '<OOV>'


# In[5]:


# Keras Tokenizer

tokenizer = Tokenizer(
    num_words=vocab_size,
    filters='“”‘’!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    #oov_token=oov_tok,
)

tokenizer.fit_on_texts(articles)
word_index = tokenizer.word_index

#dict(list(word_index.items())[0:10])
print(len(tokenizer.word_index))

article_sequences = tokenizer.texts_to_sequences(articles)

padded_sequences = pad_sequences(article_sequences, maxlen=max_length, padding='post', truncating='post')

print(articles[0])
print(article_sequences[0])
print(padded_sequences[0])


# In[6]:


set(labels)


# In[7]:


label_encoder = LabelEncoder()
label_encoder.fit(labels)

labels_enc = label_encoder.transform(labels)


# In[8]:


xt=np.array(padded_sequences)
yt=np.array(labels_enc)


# In[9]:


# Sequential Embedding

model = models.Sequential([
    layers.Embedding(len(tokenizer.word_index)+1, embedding_dim),
    #layers.Bidirectional(layers.LSTM(64)),
    #layers.Bidirectional(layers.GRU(64)),
    layers.GRU(embedding_dim),
    layers.Dense(len(label_encoder.classes_), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
model.summary()


# In[10]:


#callback = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.1, patience=5)

# Train the Model

history = model.fit(
    xt,
    yt,
    epochs=10, 
    verbose=True, 
    #batch_size=32, 
    validation_split=0.25,
    #callbacks=[callback]
)

# Evaluate the model

loss, accuracy = model.evaluate(xt, yt, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))


# In[ ]:


pd.DataFrame(history.history)[['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
#pd.DataFrame(history.history)[['loss', 'val_loss']].plot()


# In[ ]:


# Save the model

model.save('detect-author-seq-lstm.keras')


# In[ ]:


set(labels)


# In[ ]:


# Test the model

test_texts = [
    "So we beat on, boats against the current, borne back ceaselessly into the past."
]

print(test_texts)

test_sequences = tokenizer.texts_to_sequences(test_texts)

pred = model.predict(np.array(test_sequences))

print(pred)
print(np.argmax(pred))

pred_label_index = np.argmax(pred)
pred_label = label_encoder.classes_[pred_label_index]

print(pred_label)

