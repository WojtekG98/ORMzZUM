import csv
import random
import numpy as np
from scipy import spatial

N = 720


def training_data(file, vel="lin", bits_to_roll=20):
    file = open(file)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        lengths = []
        for i in range(1, len(row)):
            value = 0
            if i == 1:
                value = float(row[i][2:])
            elif i == len(row) - 1:
                value = float(row[i][:-1])
            else:
                value = float(row[i])
            if value == float('inf'):
                value = 0
            lengths.append(value)
        rows.append([lengths, float(row[0])])
    if len(rows) % 2 != 0:
        rows.pop()
    data = []


    for i in range(0, len(rows) - 1, 1):
        rows_0 = rows[i]  # rows[2 * i]
        rows_1 = rows[i + 1]  # rows[2 * i + 1]
        lengths_0 = rows_0[0]
        lengths_1 = rows_1[0]
        lengths_diff = []
        if vel == "lin":
            for j in range(0, len(lengths_0)):
                lengths_diff.append(lengths_1[j] - lengths_0[j])
        if vel == "ang":
            rolled_lengths_0 = np.roll(lengths_0, bits_to_roll)
            for j in range(0, len(lengths_0)):
                lengths_diff.append(lengths_1[j] - rolled_lengths_0[j])
        vel_0 = rows_0[1]
        vel_1 = rows_1[1]
        vel_mean = vel_1  # (vel_0 + vel_1) / 2
        data.append([lengths_diff, vel_mean])
    print(len(rows), len(data))
    return data


if __name__ == "__main__":
    workspace_path = 'raw_data/room_to_room_saved_data/'
    save_path = 'room_to_room/data/'
    vels = ("ang", "lin")
    split = False
    for vel in vels:
        filename = workspace_path + vel + '.csv'
        all_data = training_data(filename, vel, 1)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        num_of_samples = 8
        i = [x for x in range(0, N, int(N / num_of_samples))]
        if split:
            random.shuffle(all_data)
            test_data = np.array(all_data[4 * len(all_data) // 5:])
            train_data = np.array(all_data[:4 * len(all_data) // 5])
            test_data_8 = []
            train_data_8 = []
            for item in train_data:
                x = [item[0][index+10] for index in i]
                train_data_8.append((x, item[1]))
            for item in test_data:
                x = [item[0][index+10] for index in i]
                test_data_8.append((x, item[1]))
            np.save(save_path + '8train_'+vel+'.npy', train_data_8)
            np.save(save_path + '8test_'+vel+'.npy', test_data_8)
        else:
            all_data8 = []
            for item in all_data:
                x = [item[0][index] for index in i]
                all_data8.append((x, item[1]))
            np.save(save_path + '8' + vel + '.npy', all_data8)
