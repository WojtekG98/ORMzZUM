import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input
from generate_data import training_data

# load the data
data = training_data('raw_data/ang_02_03_2022_14_34_06.csv')
X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)
X = keras.utils.normalize(X)  # scales data between 0 and 1
X = np.array(X)
Y = np.array(Y)
input_shape = X.shape[1]
model = Sequential(
    [
        Input(shape=input_shape),
        Dense(128, input_dim=input_shape, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

model.summary()

batch_size = 128
epochs = 3 # 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['mean_absolute_error'])

model.fit(X, Y, batch_size=batch_size, epochs=epochs)
model.save('linear_vel_regress.model')

#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])
