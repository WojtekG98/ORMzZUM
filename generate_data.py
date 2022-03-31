import csv
import random
import numpy as np

N = 720


def training_data(file, n=2):
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

    # for i in range(0, int(len(rows) / 2)):
    i = 0
    while i + n - 1 < len(rows):
        rows_0 = rows[i]  # rows[2 * i]
        rows_1 = rows[i + n - 1]  # rows[2 * i + 1]
        lengths_0 = rows_0[0]
        lengths_1 = rows_1[0]
        # rolled_lengths_1 = array_new = np.roll(lengths_1, 360)
        lengths_diff = []
        for j in range(0, len(lengths_0)):
            lengths_diff.append(lengths_1[j] - lengths_0[j])
        vel_0 = rows_0[1]
        vel_1 = rows_1[1]
        vel_mean = vel_1  # (vel_0 + vel_1) / 2
        data.append([lengths_diff, vel_mean])
        i += n
    print(len(rows), len(data))
    return data


if __name__ == "__main__":
    filename = 'lin_22_03_2022_18_27_55'
    data = training_data('raw_data/' + filename + '.csv', 2)
    random.shuffle(data)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    test_data = np.array(data[4 * len(data) // 5:])
    train_data = np.array(data[:4 * len(data) // 5])
    n = 8
    i = [x for x in range(0, N, int(N / n))]
    test_data_8 = []
    train_data_8 = []
    for item in train_data:
        x = [item[0][index] for index in i]
        train_data_8.append((x, item[1]))
    for item in test_data:
        x = [item[0][index] for index in i]
        test_data_8.append((x, item[1]))
    np.save('data/train_' + filename, train_data_8)
    np.save('data/test_' + filename, test_data_8)
