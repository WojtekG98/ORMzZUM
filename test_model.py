import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('linear_vel_regress.model')
filename = 'lin_22_03_2022_18_27_55'
data = np.load('data/test_' + filename + '.npy', allow_pickle=True) # training_data('raw_data/lin_02_03_2022_14_34_06.csv')
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
print(len(x_test), len(x_test[0]), len(y_test), y_test[0])
predictions = new_model.predict(x_test)
for i in range(0, len(predictions)):
    print(predictions[i], y_test[i])