import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('mnist_keras_tutorial.model')
#new_model = tf.keras.models.load_model('epic_num_reader.model')
mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

predictions = new_model.predict(x_test)
print(np.argmax(predictions[1]))
plt.imshow(x_test[1],cmap=plt.cm.binary)
plt.show()