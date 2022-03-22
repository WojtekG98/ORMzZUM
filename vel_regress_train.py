import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

vel = "linear"


# vel = "angular"


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MAPE]')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(history.history)


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(16, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        #        layers.Dropout(0.5),
        layers.Dense(8, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        #        layers.Dropout(0.5),
        layers.Dense(1, activation="linear")
    ])
    model.compile(loss='mean_absolute_percentage_error',
                  optimizer="adam")
    return model


# load the data
filename = 'lin_22_03_2022_18_27_55'
data = np.load('data/train_' + filename + '.npy', allow_pickle=True)
X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)
X = np.array(X)
Y = np.array(Y)
X_train = X
Y_train = Y
model = build_and_compile_model(layers.Normalization(axis=-1))

his = model.fit(X_train,
                Y_train,
                validation_split=0.2,
                epochs=50)
model.summary()
plot_loss(his)
model.save(vel + "_vel_regress.model")
