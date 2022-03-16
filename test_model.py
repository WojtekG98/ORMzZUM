import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from generate_data import training_data

new_model = tf.keras.models.load_model('linear_vel_regress.model')
#new_model = tf.keras.models.load_model('epic_num_reader.model')
#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(x_train, y_train), (x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
data = training_data('raw_data/lin_02_03_2022_14_34_06.csv')
X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)
# X = keras.utils.normalize(X)  # scales data between 0 and 1
X = np.array(X)
Y = np.array(Y)
x_test = X
y_test = Y
print(len(x_test), len(x_test[0]))
predictions = new_model.predict(x_test)
for i in range(0, len(predictions)):
    print(predictions[i], y_test[i])