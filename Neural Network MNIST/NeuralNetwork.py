import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(42)
num_classes = 10


with open("mnist.pkl", 'rb') as f:
    mnist = pickle.load(f, encoding='latin1')
train_set, valid_set, test_set = mnist
train_images, train_labels = train_set
valid_images, valid_labels = valid_set
test_images, test_labels = test_set

def transform_to_onehot(labels):
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


y = transform_to_onehot(train_labels)
y_valid = transform_to_onehot(valid_labels)

epochs = 10
batch_size = 1000
iters_per_epoch = int(train_images.shape[0] / batch_size)
learning_rate = 10
beta = 0.9
hidden_layer_size = 32

W1 = np.random.normal(0, 0.001, (hidden_layer_size, train_images.shape[1]))
b1 = np.zeros(hidden_layer_size)
W2 = np.random.normal(0, 0.001, (num_classes, hidden_layer_size))
b2 = np.zeros(num_classes)

VW1 = np.zeros((hidden_layer_size, train_images.shape[1]))
Vb1 = np.zeros(hidden_layer_size)
VW2 = np.zeros((num_classes, hidden_layer_size))
Vb2 = np.zeros(num_classes)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(x):
    z1 = np.matmul(x, W1.T) + b1
    a1 = relu(z1)
    z2 = np.matmul(a1, W2.T) + b2
    a2 = softmax(z2)

    cache = (x, a1, a2)
    return a2, cache


def cross_entropy_loss(y_pred, y_true):
    logprobs = y_true * np.log(y_pred)
    cost = - (1.0 / y_true.shape[0]) * np.sum(logprobs)
    return cost


def relu_backward(x):
    x[x <= 0.0] = 0.0
    x[x > 0.0] = 1.0
    return x


def accuracy(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = ((y_pred == y_true).sum() * 100) / y_true.shape[0]
    return accuracy


def compute_grads(cache, y):
    x, a1, a2 = cache

    delta2 = (1.0 / train_images.shape[0]) * (a2 - y)
    dw2 = np.matmul(delta2.T, a1)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.matmul(delta2, W2) * relu_backward(a1)

    dw1 = np.matmul(delta1.T, x)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    return np.array([dw2, db2, dw1, db1])


if __name__ == '__main__':
    train_loss = []
    validation_loss = []

    for epoch in range(epochs):
        batch_loss = 0

        for idx in range(iters_per_epoch):
            x_batch = train_images[idx * batch_size:(idx + 1) * batch_size, :]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size, :]

            out, cache = forward(x_batch)

            loss = cross_entropy_loss(out, y_batch)
            batch_loss += loss

            grads = compute_grads(cache, y_batch)

            # momentum
            VW2 = beta * VW2 + (1 - beta) * grads[0]
            W2 = W2 - learning_rate * VW2

            Vb2 = beta * Vb2 + (1 - beta) * grads[1]
            b2 = b2 - learning_rate * Vb2

            VW1 = beta * VW1 + (1 - beta) * grads[2]
            W1 = W1 - learning_rate * VW1

            Vb1 = beta * Vb1 + (1 - beta) * grads[3]
            b1 = b1 - learning_rate * Vb1

        out, _ = forward(valid_images)
        valid_loss = cross_entropy_loss(out, y_valid)
        validation_loss.append(valid_loss)

        loss = batch_loss / iters_per_epoch
        train_loss.append(loss)
        print("Train loss at epoch {}: {}".format(epoch + 1, loss))
        print("Validation loss at epoch {}: {}".format(epoch + 1, valid_loss))

    plt.plot(range(len(train_loss)), train_loss, color='red')
    plt.plot(range(len(validation_loss)), validation_loss, color='green')

    pred, _ = forward(test_images)

    test_accuracy = accuracy(pred, test_labels)
    print("Test accuracy: {}%".format(test_accuracy))

