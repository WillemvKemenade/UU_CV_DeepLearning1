import tensorflow as tf
import numpy as np

#load in the fashion dataset which is 6000 train and 6000 test
(x_train, y_train_labels), (x_test, y_test_labels) = tf.keras.datasets.fashion_mnist.load_data()

#divide the training set into a training and validation datasets
#We take 5000 for the validation
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train_labels, y_valid_labels) = y_train_labels[5000:], y_train_labels[:5000]

print(x_valid.shape)