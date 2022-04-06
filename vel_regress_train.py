import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

vel = "linear"
pre = 'data/8train_'
extension = '.npy'

# vel = "angular"


def plot_loss(history, lossname):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MAE]')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(history.history.keys())
    datafilename = 'plot_data/' + lossname + '.csv'
    file = open(datafilename, "w")
    for key in history.history.keys():
        print(key, history.history[key])
        values = history.history[key]
        line = str(key) + ', '
        for i in range(len(values)-1):
            line += str(values[i]) + ', '
        line += str(values[-1]) + '\n'
        file.write(line)
    file.close()


def build_and_compile_model():
    model = keras.Sequential([
        layers.Normalization(axis=-1),
        layers.Dense(4, activation='relu'),  # kernel_regularizer=regularizers.l2(0.0001),
        layers.Normalization(axis=-1),
        # layers.Dropout(0.1),
        #layers.BatchNormalization(),
        layers.Dense(12, activation='relu'),
        layers.Normalization(axis=-1),
        layers.Dense(4, activation='relu'),
        layers.Normalization(axis=-1),
        layers.Dense(1, activation="linear")
    ])
    model.compile(loss="mean_absolute_error",# "mean_squared_error",# "mean_squared_logarithmic_error",
                  optimizer="adam",
                  metrics=[keras.metrics.MeanSquaredError(),
                           keras.metrics.MeanSquaredLogarithmicError()])
    return model


def transform_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == "__main__":
    filename = pre + 'lin_22_03_2022_18_27_55' + extension
    X_train, Y_train = transform_data(np.load(filename, allow_pickle=True))
    regression_model = build_and_compile_model()
    his = regression_model.fit(X_train,
                               Y_train,
                               validation_split=0.2,
                               epochs=50)
    regression_model.summary()
    plot_loss(his, "mean_absolute_error")
    regression_model.save(vel + "_vel_regress.model")
