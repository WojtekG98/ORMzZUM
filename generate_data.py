import csv


def training_data(file):
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
    for i in range(0, int(len(rows)/2)):
        rows_0 = rows[2*i]
        rows_1 = rows[2*i+1]
        lengths_0 = rows_0[0]
        lengths_1 = rows_1[0]
        lengths_diff =[]
        for j in range(0, len(lengths_0)):
            lengths_diff.append(lengths_1[j]-lengths_0[j])
        vel_0 = rows_0[1]
        vel_1 = rows_1[1]
        vel_mean = vel_0+vel_1/2
        data.append([lengths_diff, vel_mean])
    # print(len(rows), len(data))
    return data
