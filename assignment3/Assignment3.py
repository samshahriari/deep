# from functions import *
from matplotlib import pyplot as plt
import numpy as np


NUM_CLASSES = 10
NUM_IMAGES = 10000
DIMENSION = 32*32*3  # 3072
M_HIDDEN_NODES = 50
np.random.seed(2424)
hidden_nodes_per_layer = [50, 50]

eta_min = 1e-5
eta_max = 1e-1
n_s = 500
batch_size = 100


def loadBatchNP(batch_name: str) -> list[np.ndarray, np.ndarray, np.ndarray]:
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


def loadAllTrainingData():
    X, Y, y = loadBatchNP("data_batch_1")
    for i in range(2, 6):
        X_new, Y_new, y_new = loadBatchNP(f"data_batch_{i}")
        X = np.concatenate((X, X_new), axis=1)
        Y = np.concatenate((Y, Y_new), axis=1)
        y = np.concatenate((y, y_new))
    ind = np.random.permutation(X.shape[1])
    return preProcess(X[:, ind[5000:]]), Y[:, ind[5000:]], y[ind[5000:]], preProcess(X[:, ind[:5000]]), Y[:, ind[:5000]], y[ind[:5000]]


def preProcess(X: np.ndarray) -> np.ndarray:
    X -= np.mean(X, axis=1, keepdims=True)
    X /= np.std(X, axis=1, keepdims=True)
    return X


def evaluateClassifier(X: np.ndarray, W: np.ndarray, b):
    from functions import softmax
    x = [X]
    for l in range(len(hidden_nodes_per_layer)):
        s = W[l] @ x[-1] + b[l]
        x.append(np.maximum(0, s))

    s = W[-1] @ x[-1] + b[-1]

    p = softmax(s)
    return p, x


def computeLoss(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_):
    p, h = evaluateClassifier(X, W, b)
    return np.sum(-Y * np.log(p))/X.shape[1]


# vi vill minimera kostnaden för att få så bra parametrar som möjligt
def computeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_) -> float:
    reg_cost = 0
    for Wl in W:
        reg_cost += np.sum(Wl**2)
    return computeLoss(X, Y, W, b, lambda_) + lambda_ * reg_cost


def computeAccuracy(X, y, W, b):
    p, _ = evaluateClassifier(X, W, b)
    prediction = p.argmax(axis=0)
    return np.mean(prediction == y)


# gradienten talar om oss vilket håll som är den största maximeringen, negativa den blir då den största minimieringen.
def computeGradients(X: np.ndarray, Y: np.ndarray, P: np.ndarray,  W: np.ndarray, b, lambda_):
    size_batch = X[0].shape[1]
    G = -(Y - P)
    dL_dW = [None] * (len(hidden_nodes_per_layer) + 1)
    dL_db = [None] * (len(hidden_nodes_per_layer) + 1)
    for l in range(len(hidden_nodes_per_layer), 0, -1):
        # print("layer", l)
        dL_dW[l] = 1 / size_batch * G @ X[l].T + 2*lambda_*W[l]
        dL_db[l] = 1 / size_batch * G @ np.ones((size_batch, 1))
        G = W[l].T@G
        G = G * (X[l] > 0)
    dL_dW[0] = 1 / size_batch * G @ X[0].T + 2*lambda_*W[0]
    dL_db[0] = 1 / size_batch * G @ np.ones((size_batch, 1))
    return dL_dW, dL_db


def relativError(v1, v2):
    t = np.abs(v1-v2)
    n = np.maximum(1e-9, np.max(v1) + np.max(v1))
    return t/n


def evaluateGradient(X, Y, W, b):
    from functions import ComputeGradsNumSlow, ComputeGradsNum
    num_features = 10
    batchSize = [1, 10, 100]
    lambdas = [0, .1, 1]  #
    threshold = 1e-6
    for num_points in batchSize:
        for lambda_ in lambdas:
            X_batch = X[:num_features, 0:num_points]
            Y_batch = Y[:, 0:num_points]
            W_batch = [W[0][:, 0:num_features]] + W[1:]
            P, H = evaluateClassifier(X_batch, W_batch, b)
            analytical_W, analytical_b = computeGradients(H, Y_batch, P, W_batch, b, lambda_)
            slow_W, slow_b = ComputeGradsNumSlow(X_batch, Y_batch, P, W_batch, b, lambda_, 1e-6)
            for l in range(len(analytical_b)):
                abs_slow_W = np.abs(analytical_W[l] - slow_W[l])
                rel_slow_W = relativError(analytical_W[l], slow_W[l])
                abs_slow_b = np.abs(analytical_b[l] - slow_b[l])
                rel_slow_b = relativError(analytical_b[l], slow_b[l])

                print()
                print("------")
                print("layer", l)
                print("num features:", num_features, ", batch size:", num_points, ", lambda: ", lambda_)
                print("SLOW W", "abs wrong", np.sum(abs_slow_W > threshold), "of", len(abs_slow_W.flatten()), "biggest error", np.max(abs_slow_W))
                print("SLOW W", "rel wrong", np.sum(rel_slow_W > threshold), "of", len(rel_slow_W.flatten()), "biggest error", np.max(rel_slow_W))
                print("SLOW b", "abs wrong", np.sum(abs_slow_b > threshold), "of", len(abs_slow_b.flatten()), "biggest error", np.max(abs_slow_b))
                print("SLOW b", "rel wrong", np.sum(rel_slow_b > threshold), "of", len(rel_slow_b.flatten()), "biggest error", np.max(rel_slow_b))


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
            grad_W, grad_b = computeGradients(H_batch, Y_batch, P_batch, W, b, lambda_)
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


def miniBatchGDCyclic(X, Y, y, lambda_, X_val, Y_val, y_val, l_cycles, n_s, W=None, b=None, plotFig=False):
    if W == None:
        W, b = initialize_weight_bias()

    eta_min = 1e-5
    eta_max = 1e-1
    batch_size = 100
    cost_training = []
    cost_val = []
    loss_training = []
    loss_val = []
    accuracy_training = []
    accuracy_val = []
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
            grad_W, grad_b = computeGradients(H_batch, Y_batch, P_batch,  W, b, lambda_)
            for i in range(len(W)):
                W[i] -= eta*grad_W[i]
                b[i] -= eta*grad_b[i]

        # print("epoch", epoch, ", cost", computeCost(X, Y, W, b, lambda_))
            t += 1
            if plotFig and (t % (2*n_s/10) == 0 or t == 1):
                cost_training.append(computeCost(X, Y, W, b, lambda_))
                cost_val.append(computeCost(X_val, Y_val, W, b, lambda_))
                loss_training.append(computeLoss(X, Y, W, b, lambda_))
                loss_val.append(computeLoss(X_val, Y_val, W, b, lambda_))
                accuracy_training.append(computeAccuracy(X, y, W, b))
                accuracy_val.append(computeAccuracy(X_val, y_val, W, b))
                x_axes.append(t)
            if t % (2*n_s) == 0:
                l += 1

    if plotFig:
        plot(x_axes, cost_training, cost_val, "cost", n_s, l_cycles)
        plot(x_axes, loss_training, loss_val, "loss", n_s, l_cycles)
        plot(x_axes, accuracy_training, accuracy_val, "accuracy", n_s, l_cycles)

    return W, b


def plot(x, y_train, y_val, title: str, n_s, l):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
    plt.clf()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(n_s))
    plt.xlabel('update step')
    plt.ylabel(title)
    plt.plot(x, y_train, label=f"{title} training")
    plt.plot(x, y_val, label=f"{title} validation")
    plt.legend()
    plt.ylim(bottom=0)
    plt.title(f"{title.capitalize()} plot")
    plt.savefig(f'results/{l}-{n_s}{title}.png')
    plt.savefig(f'results/{l}-{n_s}{title}.pgf')
    # plt.show()


def initialize_weight_bias(dim=DIMENSION, m=M_HIDDEN_NODES):
    W = []
    b = []
    W.append(1/np.sqrt(dim)*np.random.randn(hidden_nodes_per_layer[0], dim))
    b.append(np.zeros((hidden_nodes_per_layer[0], 1)))
    for l in range(1, len(hidden_nodes_per_layer)):
        W.append(1/np.sqrt(hidden_nodes_per_layer[l-1]) * np.random.randn(hidden_nodes_per_layer[l], hidden_nodes_per_layer[l-1]))
        b.append(np.zeros((hidden_nodes_per_layer[l], 1)))
    W.append(1/np.sqrt(hidden_nodes_per_layer[-1])*np.random.randn(NUM_CLASSES, hidden_nodes_per_layer[-1]))
    b.append(np.zeros((NUM_CLASSES, 1)))
    return W, b


def lambdaSearch(l_min, l_max, testing_points, cycles):
    X_train, Y_train, y_train, X_val, Y_val, y_val = loadAllTrainingData()
    n_batch = 100
    n_s = 2 * np.floor(X_train.shape[1]/n_batch)
    l = l_min + (l_max - l_min) * np.random.rand(testing_points)
    lambdas = np.power(10, l)

    res = []
    for lambda_ in lambdas:

        W, b = miniBatchGDCyclic(X_train, Y_train, y_train, lambda_, X_val, Y_val, y_val, cycles, n_s)
        acc = computeAccuracy(X_val, y_val, W, b)
        res.append((acc, lambda_))
    res.sort(reverse=True)
    for (acc, lam) in res:
        print("Accuracy", acc, "lambda", lam, "log lambda", np.log10(lam))
    return res[0][1]


def main():
    X_train, Y_train, y_train = loadBatchNP("data_batch_1")
    X_val, Y_val, y_val = loadBatchNP("data_batch_2")
    X_test, Y_test, y_test = loadBatchNP("test_batch")
    X_train = preProcess(X_train)
    X_val = preProcess(X_val)
    X_test = preProcess(X_test)
    # W, b = initialize_weight_bias()
    # print("Hidden layer", hidden_nodes_per_layer)
    # evaluateGradient(X_train, Y_train, W, b)
    # W, b = initialize_weight_bias()
    # for i in range(len(W)):
    #     print("W", W[i].shape)
    #     print("b", b[i].shape)

    miniBatchGDCyclic(X_train, Y_train, y_train, .01, X_val, Y_val, y_val, 1, 500, plotFig=True)
    miniBatchGDCyclic(X_train, Y_train, y_train, .01, X_val, Y_val, y_val, 3, 800, plotFig=True)

    X_train, Y_train, y_train, X_val, Y_val, y_val = loadAllTrainingData()
    best_lambda = lambdaSearch(-1, -5, 8, 2)
    print(np.log10(best_lambda))
    best_lambda = lambdaSearch(np.log10(best_lambda)+1, np.log10(best_lambda)-1, 20, 2)
    print(np.log10(best_lambda))
    W, b = miniBatchGDCyclic(X_train, Y_train, y_train, best_lambda, X_val, Y_val, y_val, 3, 1800, plotFig=True)
    print(computeAccuracy(X_test, y_test, W, b))


if __name__ == "__main__":
    main()
