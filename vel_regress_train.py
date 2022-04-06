import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

vel = "linear"


# vel = "angular"


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MAE]')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(history.history.keys())


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(4, activation='relu'),  # kernel_regularizer=regularizers.l2(0.0001),
        # layers.Dropout(0.1),
        #layers.BatchNormalization(),
        layers.Dense(12, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation="linear")
    ])
    model.compile(loss="mean_absolute_error",# "mean_squared_error",# "mean_squared_logarithmic_error",
                  optimizer="adam",
                  metrics=[keras.metrics.MeanSquaredError(),
                           keras.metrics.MeanSquaredLogarithmicError()])
    return model


# load the data
filename = 'lin_22_03_2022_18_27_55'
data = np.load('data/8train_' + filename + '.npy', allow_pickle=True)
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
                epochs=10)
model.summary()
plot_loss(his)
model.save(vel + "_vel_regress.model")
