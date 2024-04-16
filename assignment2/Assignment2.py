# from functions import *
from matplotlib import pyplot as plt
import numpy as np


NUM_CLASSES = 10
NUM_IMAGES = 10000
DIMENSION = 32*32*3  # 3072
M_HIDDEN_NODES = 50
np.random.seed(2424)

eta_min = 1e-5
eta_max = 1e-1
n_s = 500
batch_size = 100


def loadBatchNP(batch_name: str):
    from functions import LoadBatch
    """
    - X is d*n
    - Y is k*n
    - y is n
    """
    A = LoadBatch(batch_name)
    X = np.asarray(A[b'data'], dtype=float).T
    y = np.asarray(A[b'labels'])
    Y = np.zeros((NUM_CLASSES, NUM_IMAGES))
    for i, k in enumerate(y):
        Y[k, i] = 1
    return X, Y, y


def preProcess(X: np.ndarray) -> np.ndarray:
    X -= np.mean(X, axis=1, keepdims=True)
    X /= np.std(X, axis=1, keepdims=True)
    return X


def evaluateClassifier(X: np.ndarray, W: np.ndarray, b):
    from functions import softmax
    # print(W[0].shape, W[1].shape, X.shape)
    s1 = W[0] @ X + b[0]
    # print(s1.shape)
    h = np.maximum(0, s1)
    # print(h.shape)
    # print(W[1].shape)
    s = W[1] @ h + b[1]
    p = softmax(s)
    return p, h


def computeLoss(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_):
    p, h = evaluateClassifier(X, W, b)
    return np.sum(-Y * np.log(p))/X.shape[1]


# vi vill minimera kostnaden för att få så bra parametrar som möjligt
def computeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_) -> float:
    return computeLoss(X, Y, W, b, lambda_) + lambda_ * (np.sum(W[0]**2) + np.sum(W[1]**2))  # want to multiply element wise


def computeAccuracy(X, y, W, b):
    p, h = evaluateClassifier(X, W, b)
    prediction = p.argmax(axis=0)
    return np.mean(prediction == y)


# gradienten talar om oss vilket håll som är den största maximeringen, negativa den blir då den största minimieringen.
def computeGradients(X: np.ndarray, Y: np.ndarray, P: np.ndarray, H,  W: np.ndarray, b, lambda_):
    size_batch = X.shape[1]
    G = -(Y - P)

    dL_dW2 = 1 / size_batch * G @ H.T + 2*lambda_*W[1]
    dL_db2 = 1 / size_batch * G @ np.ones((size_batch, 1))
    G = W[1].T@G
    G = G * (H > 0)
    dL_dW1 = 1 / size_batch * G @ X.T + 2*lambda_*W[0]
    dL_db1 = 1 / size_batch * G @ np.ones((size_batch, 1))
    return [dL_dW1, dL_dW2], [dL_db1, dL_db2]


def relativError(v1, v2):
    t = np.abs(v1-v2)
    n = np.maximum(1e-9, np.max(v1) + np.max(v1))
    return t/n


def evaluateGradient(X, Y, W, b):
    from functions import ComputeGradsNumSlow, ComputeGradsNum
    num_features = 20
    batchSize = [1, 10, 100]
    lambdas = [0, .1, 1]
    threshold = 1e-6
    for num_points in batchSize:
        for lambda_ in lambdas:
            X_batch = X[:num_features, 0:num_points]
            Y_batch = Y[:, 0:num_points]
            W_batch = [W[0][:, 0:num_features], W[1]]
            P, H = evaluateClassifier(X_batch, W_batch, b)
            analytical_W, analytical_b = computeGradients(X_batch, Y_batch, P, H, W_batch, b, lambda_)
            slow_W, slow_b = ComputeGradsNumSlow(X_batch, Y_batch, P, W_batch, b, lambda_, 1e-6)

            abs_slow_W = np.abs(analytical_W[1] - slow_W[1])
            rel_slow_W = relativError(analytical_W[1], slow_W[1])
            abs_slow_b = np.abs(analytical_b[1] - slow_b[1])
            rel_slow_b = relativError(analytical_b[1], slow_b[1])

            print("------")
            print("num features:", num_features, ", batch size:", num_points, ", lambda: ", lambda_)
            print("SLOW W", "abs wrong", np.sum(abs_slow_W > threshold), "biggest error", np.max(abs_slow_W))
            print("SLOW W", "rel wrong", np.sum(rel_slow_W > threshold), "biggest error", np.max(rel_slow_W))
            print("SLOW b", "abs wrong", np.sum(abs_slow_b > threshold), "biggest error", np.max(abs_slow_b))
            print("SLOW b", "rel wrong", np.sum(rel_slow_b > threshold), "biggest error", np.max(rel_slow_b))


def miniBatchGD(X, Y, n_batch, eta, n_epochs, W, b, lambda_, X_test, Y_test):
    import matplotlib.pyplot as plt
    cost_training = []
    cost_test = []
    loss_training = []
    loss_test = []
    for epoch in range(n_epochs):
        ind = np.random.permutation(X.shape[1])
        # print(M[:, ind[:3]])
        for j in range(0, X.shape[1], n_batch):
            j_start = j
            j_end = j+n_batch
            X_batch = X[:, ind[j_start:j_end]]
            Y_batch = Y[:, ind[j_start:j_end]]
            P_batch, H_batch = evaluateClassifier(X_batch, W, b)
            grad_W, grad_b = computeGradients(X_batch, Y_batch, P_batch, H_batch, W, b, lambda_)
            for i in range(len(W)):
                W[i] -= eta*grad_W[i]
                b[i] -= eta*grad_b[i]
        # print("epoch", epoch, ", cost", computeCost(X, Y, W, b, lambda_))
        cost_training.append(computeCost(X, Y, W, b, lambda_))
        cost_test.append(computeCost(X_test, Y_test, W, b, lambda_))
        loss_training.append(computeLoss(X, Y, W, b, lambda_))
        loss_test.append(computeLoss(X_test, Y_test, W, b, lambda_))
    plt.plot(cost_training, label="cost training")
    plt.plot(loss_training, label="loss training", linestyle=(0, (5, 10)))
    plt.plot(cost_test, label="cost validation")
    plt.plot(loss_test, label="loss validation", linestyle=(0, (5, 10)))
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.title(f"n_batch={n_batch}, eta={eta}, n_epochs={n_epochs}, lambda={lambda_}")
    plt.grid(True)
    plt.show()

    return W, b


def miniBatchGDCyclic(X, Y, y, W, b, lambda_, X_test, Y_test, y_test, l_cycles, n_s, plot=False):

    eta_min = 1e-5
    eta_max = 1e-1
    batch_size = 100
    cost_training = []
    cost_test = []
    loss_training = []
    loss_test = []
    accuracy_training = []
    accuracy_test = []
    x_axes = []
    eta = eta_min
    l = 0
    t = 0
    # for epoch in range(n_epochs):
    while l < l_cycles:
        ind = np.random.permutation(X.shape[1])
        # print(M[:, ind[:3]])
        for j in range(0, X.shape[1], batch_size):
            if 2*l*n_s <= t and t <= (2*l+1)*n_s:
                eta = eta_min + (t-2*l*n_s)/n_s*(eta_max-eta_min)
            else:
                eta = eta_max - (t-(2*l+1)*n_s)/n_s*(eta_max-eta_min)

            j_start = j
            j_end = j+batch_size
            X_batch = X[:, ind[j_start:j_end]]
            Y_batch = Y[:, ind[j_start:j_end]]
            P_batch, H_batch = evaluateClassifier(X_batch, W, b)
            grad_W, grad_b = computeGradients(X_batch, Y_batch, P_batch, H_batch, W, b, lambda_)
            for i in range(len(W)):
                W[i] -= eta*grad_W[i]
                b[i] -= eta*grad_b[i]

        # print("epoch", epoch, ", cost", computeCost(X, Y, W, b, lambda_))
            t += 1
            if plot and (t % (2*n_s/10) == 0 or t == 1):
                cost_training.append(computeCost(X, Y, W, b, lambda_))
                cost_test.append(computeCost(X_test, Y_test, W, b, lambda_))
                loss_training.append(computeLoss(X, Y, W, b, lambda_))
                loss_test.append(computeLoss(X_test, Y_test, W, b, lambda_))
                accuracy_training.append(computeAccuracy(X, y, W, b))
                accuracy_test.append(computeAccuracy(X_test, y_test, W, b))
                x_axes.append(t)
            if t % (2*n_s) == 0:
                l += 1

    if plot:
        plot(x_axes, cost_training, cost_test, "cost", n_s, l_cycles)
        plot(x_axes, loss_training, loss_test, "loss", n_s, l_cycles)
        plot(x_axes, accuracy_training, accuracy_test, "accuracy", n_s, l_cycles)

    return W, b


def plot(x, y_train, y_val, title: str, n_s, l):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
    plt.clf()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(n_s/2))
    plt.xlabel('update step')
    plt.ylabel(title)
    plt.plot(x, y_train, label=f"{title} training")
    plt.plot(x, y_val, label=f"{title} validation")
    plt.legend()
    plt.ylim(bottom=0)
    plt.title(f"{title.capitalize()} plot")
    plt.savefig(f'results/{l}-{n_s}{title}.pgf')


def initialize_weight_bias(dim=DIMENSION, m=M_HIDDEN_NODES):
    W1 = 1/np.sqrt(dim)*np.random.randn(m, dim)
    b1 = np.zeros((m, 1))
    W2 = 1/np.sqrt(m)*np.random.randn(NUM_CLASSES, m)
    b2 = np.zeros((NUM_CLASSES, 1))
    W = [W1, W2]
    b = [b1, b2]
    return W, b


def main():
    X_train, Y_train, y_train = loadBatchNP("data_batch_1")
    X_val, Y_val, y_val = loadBatchNP("data_batch_2")
    X_test, Y_test, y_test = loadBatchNP("test_batch")

    X_train = preProcess(X_train)
    X_val = preProcess(X_val)
    X_test = preProcess(X_test)

    # montage(X.T)
    W, b = initialize_weight_bias()

    # print(X.shape)

    print(computeCost(X_train, Y_train, W, b, 1000))

    print(computeAccuracy(X_train, y_train, W, b))

    # evaluateGradient(X_train, Y_train, W, b)
    # W, b = miniBatchGD(X_train[:, :100], Y_train[:, :100], 100, 0.01, 200, W, b, 0, X_val, Y_val)
    W, b = miniBatchGDCyclic(X_train, Y_train, y_train, W, b, .01, X_test, Y_test, y_test, 1, 500)
    W, b = initialize_weight_bias()
    W, b = miniBatchGDCyclic(X_train, Y_train, y_train, W, b, .01, X_test, Y_test, y_test, 3, 800)
    print(computeAccuracy(X_train, y_train, W, b))


if __name__ == "__main__":
    main()
