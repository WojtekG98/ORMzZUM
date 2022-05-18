import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def plot_loss(history, lossname):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MAE]')
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
    plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(history.history['mean_squared_logarithmic_error'], label='mean_squared_logarithmic_error')
    plt.plot(history.history['val_mean_squared_logarithmic_error'], label='val_mean_squared_logarithmic_error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSLE]')
    plt.legend()
    plt.grid(True)
    plt.show()
    datafilename = 'plot_data/' + lossname + '.csv'
    file = open(datafilename, "w")
    for key in history.history.keys():
        print(key, history.history[key])
        values = history.history[key]
        line = str(key) + ', '
        for i in range(len(values) - 1):
            line += str(values[i]) + ', '
        line += str(values[-1]) + '\n'
        file.write(line)
    file.close()


def build_and_compile_model(activation_name, output_activation_name):
    model = keras.Sequential([
        layers.Normalization(axis=-1),
        layers.Dense(4, activation=activation_name),  # kernel_regularizer=regularizers.l2(0.0001),
        layers.Normalization(axis=-1),
        # layers.Dropout(0.1),
        # layers.BatchNormalization(),
        layers.Dense(12, activation=activation_name),
        layers.Normalization(axis=-1),
        layers.Dense(4, activation=activation_name),
        layers.Normalization(axis=-1),
        layers.Dense(1, activation=output_activation_name)
    ])
    model.compile(loss="mean_absolute_error",  # "mean_squared_error",# "mean_squared_logarithmic_error",
                  optimizer="Adam",
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


def train_model(vel_name, epochs_num, activation_name='relu', output_activation_name='tanh'):
    extension = '.npy'
    filename = vel_name + '_22_03_2022_18_27_55' + extension
    x_train, y_train = transform_data(np.load(filename, allow_pickle=True))
    regression_model = build_and_compile_model(activation_name, output_activation_name)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
    his = regression_model.fit(x_train,
                               y_train,
                               validation_split=0.2,
                               epochs=epochs_num,
                               callbacks=[callback],
                               verbose=1)
    regression_model.summary()
    plot_loss(his, "mean_absolute_error")
    regression_model.save(vel + "_vel_regress.model")


if __name__ == "__main__":
    vel = 'ang' # 'tanh' 'tanh'
    pre = 'data/train_' #
    train_model(pre + vel, 2500, 'tanh', 'tanh')
