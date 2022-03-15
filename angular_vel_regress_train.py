import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import layers
from generate_data import training_data


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=keras.optimizers.Adam(0.001))
    return model


# load the data
data = training_data('raw_data/ang_09_03_2022_16_40_34.csv')
X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)
# X = keras.utils.normalize(X)  # scales data between 0 and 1
X = np.array(X)
X_train = X
Y = np.array(Y)
Y_train = Y
model = build_and_compile_model(layers.Normalization(axis=-1))

his = model.fit(X_train,
                Y_train,
                validation_split=0.2,
                verbose=0,
                epochs=100)
model.summary()
plot_loss(his)
model.save('angular_vel_regress.model')

test_data = training_data('raw_data/ang_02_03_2022_14_34_06.csv')
X_test = []
Y_test = []
for features, label in test_data:
    X_test.append(features)
    Y_test.append(label)
ans = model.predict(X_test)
for i in range(0, len(ans)):
    print(ans[i], Y_test[i])
