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

