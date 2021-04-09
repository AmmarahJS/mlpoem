import tensorflow as tf
import string
import requests
import pandas as pd
from tensorflow import keras
response = requests.get('https://github.com/AmmarahJS/poemdata/blob/main/Poem%20Database%20Text.txt')
response.text
data = response.text.splitlines()
len(data)
len(" ".join(data))

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

token = Tokenizer()
token.fit_on_texts(data)

token.word_index

encoded_text = token.texts_to_sequences(data)

x = ['i am good']
token.texts_to_sequences(x)
vocab_size = len(token.word_counts) + 1

datalist = []
for d in encoded_text:
  if len(d)>1:
    for i in range(2, len(d)):
      datalist.append(d[:i])
      (d[:i])
      
max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')

X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=50)

poetry_length = 10
def generate_poetry(seed_text, n_lines):
  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      encoded = token.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')

      y_pred = np.argmax(model.predict(encoded), axis=-1)

      predicted_word = ""
      for word, index in token.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      seed_text = seed_text + ' ' + predicted_word
      text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)
    print(text)

seed_text = 'i am good'
generate_poetry(seed_text, 5)


