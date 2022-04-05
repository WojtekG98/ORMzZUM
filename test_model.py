import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    new_model = tf.keras.models.load_model('linear_vel_regress.model')
    filename = 'lin_22_03_2022_18_27_55'
    data = np.load('data/8test_' + filename + '.npy',
                   allow_pickle=True)  # training_data('raw_data/lin_02_03_2022_14_34_06.csv')
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
    print(new_model.summary())

    predictions = new_model.predict(x_test)
    i = [x for x in range(0, len(predictions))]
    plt.figure()
    plt.plot(i, predictions, 'r.', label="predictions")
    plt.plot(i, y_test, 'b.', label="labels")
    plt.legend()
    plt.show()

    diff = []
    diff_sq = []
    diff_abs = []
    for i in range(0, len(predictions)):
        diff.append(Y[i] - predictions[i])
        diff_abs.append(abs(Y[i] - predictions[i]))
        diff_sq.append((Y[i] - predictions[i])**2)
    hist, bins = np.histogram(diff, 100)
    plot_np_hist(hist, bins)
    print(np.sum(diff_sq),np.sum(diff_abs), len(predictions))
    # hist, bins = np.histogram(diff, 50)
    # plot_np_hist(hist, bins)
