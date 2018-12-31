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

# Classifications
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Interesting print statements
print(train_images.shape) # (60000, 28, 28)
print(len(train_labels))  # 60000
print(train_labels)       # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))   # 10000

# Preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# View plot
# plt.show()

# Consistent preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# View plot
# plt.show()

# Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

