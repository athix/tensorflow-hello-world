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
#plt.plot(train_data[0])
#plt.show()

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

# Create a smaller model

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])

print(smaller_model.summary())

smaller_history = smaller_model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2)

# Create a bigger model

bigger_model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(256, activation=tf.nn.relu), # Using 256 to be easier on memory
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])

print(bigger_model.summary())

# This kills the PC.
bigger_history = bigger_model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2)

#######################################
## Plot training and validation loss ##
#######################################

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(
                history.epoch,
                history.history['val_'+key],
                '--',
                label=name.title()+' Val')
        plt.plot(
                history.epoch,
                history.history[key],
                color=val[0].get_color(),
                label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

plot_history([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)])

################
## Strategies ##
################

# Add weight regularization

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

l2_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2)

plot_history([
    ('baseline', baseline_history),
    ('l2', l2_model_history)])

# Add dropout

dropout_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dropout_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])

dropout_model_history = dropout_model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2)

plot_history([
    ('baseline', baseline_history),
    ('dropout', dropout_model_history)])

# View all three strategies together

plot_history([
    ('baseline', baseline_history),
    ('l2', l2_model_history),
    ('dropout', dropout_model_history)
])
