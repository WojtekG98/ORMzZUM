import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers


def plot_loss(history, lossname):
    plt.figure()
    plt.plot(history.history['loss'], label='Funkcja kosztu na zbiorze uczącym')
    plt.plot(history.history['val_loss'], label='Funkcja kosztu na zbiorze testowym')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(history.history['mean_squared_error'], label='Funkcja kosztu na zbiorze uczącym')
    plt.plot(history.history['val_mean_squared_error'], label='Funkcja kosztu na zbiorze testowym')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(history.history['mean_squared_logarithmic_error'], label='Funkcja kosztu na zbiorze uczącym')
    plt.plot(history.history['val_mean_squared_logarithmic_error'], label='Funkcja kosztu na zbiorze testowym')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.legend()
    plt.grid(True)
    plt.show()
    file = open(lossname, "w")
    for key in history.history.keys():
        print(key, history.history[key])
        values = history.history[key]
        line = str(key) + ', '
        for i in range(len(values) - 1):
            line += str(values[i]) + ', '
        line += str(values[-1]) + '\n'
        file.write(line)
    file.close()


def build_and_compile_model(norm, activation_name, output_activation_name):
    model = keras.Sequential([
        norm,
        layers.Dense(4, activation=activation_name),  # kernel_regularizer=regularizers.l2(0.0001),
        layers.LayerNormalization(axis=-1),
        # layers.Dropout(0.1),
        # layers.BatchNormalization(),
        layers.Dense(12, activation=activation_name),
        layers.LayerNormalization(axis=-1),
        layers.Dense(4, activation=activation_name),
        layers.LayerNormalization(axis=-1),
        layers.Dense(1, activation=output_activation_name)
    ])
    model.compile(loss="mean_absolute_error",
                  optimizer="Adam",
                  metrics=[keras.metrics.MeanSquaredError(),
                           keras.metrics.MeanSquaredLogarithmicError()])
    return model


def transform_data_max(data):
    x = []
    y = []
    near_0 = 0
    for features, label in data:
        if abs(label) < 0.02:
            if near_0 < 500:
                x.append(features)
                y.append(label)
                near_0 += 1
        else:
            x.append(features)
            y.append(label)
    x = np.array(x)
    y = np.array(y)

    hist, bins = np.histogram(y, 100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    return x, y


def transform_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y


def train_model(workspace_folder, vel_name, epochs_num, activation_name='relu', output_activation_name='tanh'):
    data_filename = workspace_folder + "data/8train_" + vel_name + '_22_03_2022_18_27_55.npy'
    x_train, y_train = transform_data_max(np.load(data_filename, allow_pickle=True))
    #x_eights, y_eights = transform_data(np.load("osemki/data/8ang.npy", allow_pickle=True))
    #x_train_all = np.append(x_train, x_eights, axis=0)
    #y_train_all = np.append(y_train, y_eights, axis=0)
    layer = layers.Normalization()
    layer.adapt(x_train)
    regression_model = build_and_compile_model(layer, activation_name, output_activation_name)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    his = regression_model.fit(x_train,
                               y_train,
                               validation_split=0.2,
                               epochs=epochs_num,
                               callbacks=[callback],
                               verbose=1)
    regression_model.summary()
    plot_loss(his, workspace_folder + vel_name + "_training_loss_and_metrics.csv")
    regression_model.save(workspace_folder + vel_name + "_vel_regress.model")


if __name__ == "__main__":
    train_model('', 'ang', 500, 'relu', 'linear')
