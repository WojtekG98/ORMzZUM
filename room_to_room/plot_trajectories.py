import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf


def plot_real_trajectory(filename="data/pos.csv"):
    file = open(filename)
    csvreader = csv.reader(file)
    x = []
    y = []
    for row in csvreader:
        x.append(float(row[0]))
        y.append(float(row[1]))

    plt.figure(0)
    #plt.xlim(-10, 0)
    #plt.ylim(-5, 5)
    plt.grid(True)
    plt.plot(x, y, "r.")


def plot_integrated_trajectory(lin_vels, ang_vels, color="b."):
    x = [-16.76]
    y = [-3.32]
    theta = [0]
    T = 1/21.25#24.905
    for i in range(len(lin_vels)):
        act_theta = theta[-1] + ang_vels[i] * T
        act_x = x[-1] + lin_vels[i]*math.cos(theta[-1]) * T
        act_y = y[-1] + lin_vels[i]*math.sin(theta[-1]) * T
        x.append(act_x)
        y.append(act_y)
        theta.append(act_theta)

    plt.figure(0)
    plt.grid(True)
    plt.plot(x, y, color)
    plt.axis('square')


def integrated_true_trajectory(filenames=("data/8ang.npy", "data/8lin.npy")):
    ang_file = np.load(filenames[0], allow_pickle=True)
    lin_file = np.load(filenames[1], allow_pickle=True)
    ang_vels = []
    lin_vels = []
    for diffs, vel in ang_file:
        ang_vels.append(vel)
    for diffs, vel in lin_file:
        lin_vels.append(vel)
    plot_integrated_trajectory(lin_vels, ang_vels)


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


def integrated_estimated_trajectory(ang_model_filename="ang_vel_regress.model",
                                    lin_model_filename="lin_vel_regress.model",
                                    filenames=("data/8ang.npy", "data/8lin.npy")):
    ang_model = tf.keras.models.load_model(ang_model_filename)
    lin_model = tf.keras.models.load_model(lin_model_filename)
    ang_file = np.load(filenames[0], allow_pickle=True)
    lin_file = np.load(filenames[1], allow_pickle=True)
    ang_diffs = []
    lin_diffs = []
    ang_vels = []
    lin_vels = []
    for diffs, vel in ang_file:
        ang_diffs.append(diffs)
        ang_vels.append(vel)
    for diffs, vel in lin_file:
        lin_diffs.append(diffs)
        lin_vels.append(vel)
    ang_train_file = np.load("data/mala8ang.npy", allow_pickle=True)
    ang_train_diffs = []
    ang_train_vels = []
    for diffs, vel in ang_train_file:
        ang_train_diffs.append(diffs)
        ang_train_vels.append(vel)
    ang_model.fit(np.array(ang_train_diffs),
                  np.array(ang_train_vels),
                  validation_split=0.2,
                  epochs=30,
                  verbose=1)
    lin_diffs = np.array(lin_diffs)
    ang_diffs = np.array(ang_diffs)
    ang_predictions = ang_model.predict(ang_diffs)
    ang_float_predicitions = [float(x) for x in ang_predictions]
    lin_predictions = lin_model.predict(lin_diffs)
    lin_float_predicitions = [float(x) for x in lin_predictions]
    plot_integrated_trajectory(lin_float_predicitions, ang_float_predicitions, "g.")
    diff = []
    for i in range(0, len(ang_predictions)):
        diff.append(ang_vels[i] - ang_predictions[i])
    plt.figure()
    hist, bins = np.histogram(diff, 100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    save_to_file("ang_error_histogram_data.csv", hist, bins, "Histogram, ", "bins, ", diff)
    diff = []
    for i in range(0, len(lin_predictions)):
        diff.append(lin_vels[i] - lin_predictions[i])
    plt.figure()
    hist, bins = np.histogram(diff, 100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    save_to_file("lin_error_histogram_data.csv", hist, bins, "Histogram, ", "bins, ", diff)


if __name__ == "__main__":
    plt.figure()
    #plot_real_trajectory()
    #integrated_true_trajectory()
    integrated_estimated_trajectory()
    plt.show()
