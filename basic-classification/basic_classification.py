# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras

# Numpy & Matplot
import numpy as np
import matplotlib.pyplot as plt

# Version >= 1.12.0
print(tf.__version__)

# Download fashion dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

