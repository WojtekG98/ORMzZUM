from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


def main(path=""):
    filename = path + "error_histogram_data.csv"
    bins = []
    patches = []
    diffs = []
    for item in open(filename, 'r'):
        line = item.split(",")
        if line[0] == 'Histogram':
            line.pop(0)
            for word in line:
                patches.append(int(word))
        if line[0] == 'bins':
            line.pop(0)
            for word in line:
                bins.append(float(word))
        if line[0] == 'diffs':
            line.pop(0)
            line.pop(-1)
            for word in line:
                diffs.append(float(word))

    (mu, sigma) = norm.fit(diffs)
    print(mu, sigma)
    # the histogram of the data
    plt.figure()
    plt.hist(diffs, bins=100, color='b')
    print(len(diffs))
    plt.grid(True)
    plt.savefig(path + 'histogram.png')
    plt.figure()
    n, bins, patches = plt.hist(diffs, bins=100, density=True, alpha=0.6, color='b')
    # add a 'best fit' line
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(path+'histogram_distribution.png')
    file = open(path+"mu_sigma.txt", 'w')
    text_to_write = "mu: " + str(mu) + ", sigma: " + str(sigma)
    file.write(text_to_write)
    num = 0
    for item in diffs:
        if -0.01 < item < 0.01:
            num += 1
    text_to_write2 = "\nin delta "+str(num)+" of "+str(len(diffs))+" = "+ str(num/len(diffs)*100)+ "%"
    file.write(text_to_write2)
    file.close()


if __name__ == "__main__":
    path_to_file = "osemki/bad_lin_"
    main(path_to_file)
