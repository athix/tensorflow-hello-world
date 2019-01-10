###########
## Setup ##
###########

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # >= 1.12.0

######################
## Download Dataset ##
######################

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension)) # Why does this double enclose? Python quirk?
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0 # set specific indices of results[i] to 1s
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# View multi-hot vectors
plt.plot(train_data[0])
plt.show()

#############################
## Demonstrate Overfitting ##
#############################

# Create baseline model

baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])

print(baseline_model.summary())

baseline_history = baseline_model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2)

