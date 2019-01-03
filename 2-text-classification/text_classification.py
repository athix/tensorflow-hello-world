###########
## Setup ##
###########

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__) # >= 1.12.0

#########################
## Downloading Dataset ##
#########################

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

######################
## Explore the data ##
######################

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels))) # Training entries: 25000, labels: 25000

print(train_data[0]) # big array

print(len(train_data[0]), len(train_data[1])) # (218, 189)

# A dictionar mapping words to an integer index
word_index = imdb.get_word_index()

# The irst indices are reversed
word_index = {k:(v+3)  for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<uNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0])) # " this film was just brilliant casting [...]"

######################
## Prepare the data ##
######################

# Pad data to support conversion to integer tensor

train_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(
        test_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)

print(len(train_data[0]), len(train_data[1])) # (256, 256)

print(train_data[0]) # Big array of words in first review

#####################
## Build the model ##
#####################

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary()) # Nice little summary table

model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

#############################
## Create a validation set ##
#############################

# This is separate from the testing data, so that testing data can be used as an example of real-world testing.

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#####################
## Train the model ##
#####################

history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=40,
        batch_size=512,
        validation_data=(x_val, y_val),
        verbose=1)

