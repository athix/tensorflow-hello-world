###########
## Setup ##
###########

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__) # >= 1.12.0

######################
## Download Dataset ##
######################

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

####################
## Define a model ##
####################

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# Create a basic model instance
model = create_model()
model.summary()

#################
## Checkpoints ##
#################

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data = (test_images, test_labels),
          callbacks = [checkpoint_callback]) # pass callback to training

# Create fresh model
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Load weights from checkpoint
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Callback options
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save wieghts, every 5-epochs.
        period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [checkpoint_callback],
          validation_data = (test_images, test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)

print(latest)

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, arruacy: {:5.2f}%".format(100*acc))

