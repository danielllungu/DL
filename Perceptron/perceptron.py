import numpy as np
import matplotlib.pyplot as plt

def random_points(number_of_points):
    a = np.random.random()
    b = np.random.random()

    dataset = []

    for i in range(number_of_points):
        x = np.random.random()
        y = np.random.random()

        if y <= a * x + b:
            cls = -1
        else:
            cls = 1

        dataset.append((x, y, cls))

    np.savetxt('point.txt', dataset, fmt='%f')

def get_X_Y_w(points):
    X = points[:, :-1]
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = points[:, -1]
    w = np.zeros(3)
    return X, Y, w



def iterate(epochs, X, Y, w):
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            net = np.dot(X[i], w)
            y = 1 if net >= 0 else -1
            e = y - Y[i]
            w = w - X[i] * e

    print('w[0]: %f w[1]: %f w[2]: %f' % (w[0], w[1], w[2]))

    return w

def plot(X, Y, w):
    x1 = np.array([0, 1])
    x2 = (-x1 * w[1] - w[0]) / w[2]

    plt.plot(X[Y == 1, 1], X[Y == 1, 2], 'ro', color='red')
    plt.plot(X[Y == -1, 1], X[Y == -1, 2], 'ro', color='green')
    plt.plot(x1, x2, color='black')
    plt.axis([0, 1, 0, 1])
    plt.show()


if __name__ == '__main__':
    number_of_points=20
    random_points(number_of_points)

    points = np.genfromtxt('point.txt', delimiter=' ')
    X,Y,w=get_X_Y_w(points)

    epochs=10
    w = iterate(epochs, X, Y, w)
    plot(X,Y,w)