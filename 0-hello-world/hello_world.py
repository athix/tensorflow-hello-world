import tensorflow as tf

tf.enable_eager_execution()
print(tf.add(1, 2))

# 3

hello = tf.constant('Hello, Tensorflow!')
print(hello.numpy())

# Hello, Tensorflow!

