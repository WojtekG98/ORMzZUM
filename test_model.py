import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vel_regress_train import transform_data
from keras.utils.vis_utils import plot_model


def plot_np_hist(hist, bins, fignum=None):
    if fignum is not None:
        plt.figure(fignum)
    else:
        plt.figure()
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def save_to_file(filename, histogram_values, bins_values, title_hist=' ', title_bins=' ', diffs=[]):
    file = open(filename, "w")
    line = title_hist
    for index in range(len(histogram_values) - 1):
        line += str(histogram_values[index]) + ', '
    line += str(histogram_values[-1]) + '\n'
    file.write(line)
    line = title_bins
    for index in range(len(bins_values) - 1):
        line += str(bins_values[index]) + ', '
    line += str(bins_values[-1]) + '\n'
    file.write(line)
    line = ''.join([str(x[0]) + ", " for x in diffs])
    file.write("diffs, " + line)
    file.close()


def plot_the_model(_model, _filename):
    plot_model(
        _model,
        to_file=_filename,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=192,
        layer_range=None,
        show_layer_activations=True,
    )


if __name__ == "__main__":
    workspace_path = 'rolled_20/'
    pre = 'rolled_20/data/8test_'
    vel = 'ang'  # lin or ang
    model_filename = workspace_path + vel + '_vel_regress.model'
    saved_filename = workspace_path + vel + '_error_histogram_data.csv'
    new_model = tf.keras.models.load_model(model_filename)
    x_test, y_test = transform_data(np.load(pre + vel + '.npy', allow_pickle=True))
    #plot_the_model(new_model, vel + '_vel_regress_model.png')
    predictions = new_model.predict(x_test)
    diff = []
    for i in range(0, len(predictions)):
        diff.append(y_test[i] - predictions[i])
    hist, bins = np.histogram(diff, 100)
    plot_np_hist(hist, bins)
    save_to_file(saved_filename, hist, bins, "Histogram, ", "bins, ", diff)
    hist, bins = np.histogram(predictions, 100)
    plot_np_hist(hist, bins)
    hist, bins = np.histogram(y_test, 100)
    plot_np_hist(hist, bins)
    #file = open('data/osemki/'+vel+'_list.csv', "w")
    #line = ''.join([str(x[0]) + ", " for x in predictions])
    #file.write("predictions, " + line)
    #file.close()
