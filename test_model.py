import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vel_regress_train import transform_data, extension


def plot_np_hist(hist, bin, fignum=None):
    if fignum is not None:
        plt.figure(fignum)
    else:
        plt.figure()
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    title = "Histogram, bins:" + str(len(bins) - 1)
    plt.title(title)
    plt.show()


def save_to_file(filename, hist, bins, title=' '):
    file = open(filename, "w")
    file.write(title + '\n')
    line = ' '
    for index in range(len(hist) - 1):
        line += str(hist[index]) + ', '
    line += str(hist[-1]) + '\n'
    file.write(line)
    line = ' '
    for index in range(len(bins) - 1):
        line += str(bins[index]) + ', '
    line += str(bins[-1]) + '\n'
    file.write(line)
    file.close()


if __name__ == "__main__":
    pre = 'data/8test_'
    new_model = tf.keras.models.load_model('linear_vel_regress.model')
    data = np.load(pre + 'lin_22_03_2022_18_27_55' + extension, allow_pickle=True)
    x_test, y_test = transform_data(data)
    print(new_model.summary())

    predictions = new_model.predict(x_test)

    diff = []
    diff_sq = []
    diff_abs = []
    for i in range(0, len(predictions)):
        diff.append(y_test[i] - predictions[i])
        diff_abs.append(abs(y_test[i] - predictions[i]))
        diff_sq.append((y_test[i] - predictions[i])**2)
    hist, bins = np.histogram(diff, 100)
    plot_np_hist(hist, bins)
    save_to_file("plot_data/predictions_lin.csv", hist, bins, "Histogram, bins:" + str(len(bins) - 1))
    # hist, bins = np.histogram(diff, 50)
    # plot_np_hist(hist, bins)
