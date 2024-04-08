# from functions import *
from matplotlib import pyplot as plt
import numpy as np


NUM_CLASSES = 10
NUM_IMAGES = 10000
DIMENSION = 32*32*3  # 3072
np.random.seed(2424)


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


def evaluateClassifier(X: np.ndarray, W: np.ndarray, b) -> np.ndarray:
    from functions import softmax
    # print(W.shape, X.shape)
    s = W @ X + b
    p = softmax(s)
    return p


def computeLoss(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_):
    p = evaluateClassifier(X, W, b)
    return np.sum(-Y * np.log(p))/X.shape[1]


# vi vill minimera kostnaden för att få så bra parametrar som möjligt
def computeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_) -> float:
    return computeLoss(X, Y, W, b, lambda_) + lambda_ * np.sum(W**2)  # want to multiply element wise


def computeAccuracy(X, y, W, b):
    p = evaluateClassifier(X, W, b)
    prediction = p.argmax(axis=0)
    return np.sum(prediction == y) / len(y)


# gradienten talar om oss vilket håll som är den största maximeringen, negativa den blir då den största minimieringen.
def computeGradients(X: np.ndarray, Y: np.ndarray, P: np.ndarray, W: np.ndarray, b, lambda_):
    size_batch = np.shape(X)[1]
    G = -(Y - P)
    dJ_dW = 1 / size_batch * G @ X.T + 2*lambda_*W
    dJ_db = 1 / size_batch * G @ np.ones((size_batch, 1))
    return dJ_dW, dJ_db


def relativError(v1, v2):
    t = np.abs(v1-v2)
    n = np.maximum(1e-9, np.max(v1) + np.max(v1))
    return t/n


def evaluateGradient(X, Y, W, b):
    from functions import ComputeGradsNum, ComputeGradsNumSlow
    featureSize = [20,  500, DIMENSION]
    batchSize = [1, 10, 100]
    lambdas = [0, .1, 1]
    threshold = 1e-6
    for num_features in featureSize:
        for num_points in batchSize:
            for lambda_ in lambdas:
                X_batch = X[:num_features, 0:num_points]
                Y_batch = Y[:, 0:num_points]
                W_batch = W[:, 0:num_features]
                P = evaluateClassifier(X_batch, W_batch, b)
                analytical_W, analytical_b = computeGradients(X_batch, Y_batch, P, W_batch, b, lambda_)
                slow_W, slow_b = ComputeGradsNumSlow(X_batch, Y_batch, P, W_batch, b, lambda_, 1e-6)
                # fast_W, fast_b = ComputeGradsNum(X_batch, Y_batch, P, W_batch, b, lambda_, 1e-6)

                abs_slow_W = np.abs(analytical_W - slow_W)
                rel_slow_W = relativError(analytical_W, slow_W)
                abs_slow_b = np.abs(analytical_b - slow_b)
                rel_slow_b = relativError(analytical_b, slow_b)

                # abs_fast_W = np.abs(analytical_W - fast_W)
                # rel_fast_W = relativError(analytical_W, fast_W)
                # abs_fast_b = np.abs(analytical_b - fast_b)
                # rel_fast_b = relativError(analytical_b, fast_b)

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
        ind = np.random.permutation(NUM_IMAGES)
        # print(M[:, ind[:3]])
        for j in range(0, NUM_IMAGES, n_batch):
            j_start = j
            j_end = j+n_batch
            X_batch = X[:, ind[j_start:j_end]]
            Y_batch = Y[:, ind[j_start:j_end]]
            P_batch = evaluateClassifier(X_batch, W, b)
            grad_W, grad_b = computeGradients(X_batch, Y_batch, P_batch, W, b, lambda_)
            W -= eta*grad_W
            b -= eta*grad_b
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
    # plt.show()

    return W, b


def main():
    X_train, Y_train, y_train = loadBatchNP("data_batch_1")
    X_val, Y_val, y_val = loadBatchNP("data_batch_2")
    X_test, Y_test, y_test = loadBatchNP("test_batch")

    X_train = preProcess(X_train)
    X_val = preProcess(X_val)
    X_test = preProcess(X_test)

    # montage(X.T)

    W1 = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
    b1 = 0.01*np.random.randn(NUM_CLASSES, 1)
    W2 = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
    b2 = 0.01*np.random.randn(NUM_CLASSES, 1)
    W3 = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
    b3 = 0.01*np.random.randn(NUM_CLASSES, 1)
    W4 = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
    b4 = 0.01*np.random.randn(NUM_CLASSES, 1)

    # print(X.shape)

    # print(computeCost(X, Y, W, b, 1000))

    # print(computeAccuracy(X, y, W, b))

    # evaluateGradient(X, Y, W, b)

    W1, b1 = miniBatchGD(X_train, Y_train, 100, .1, 40, W1, b1, 0, X_val, Y_val)
    W2, b2 = miniBatchGD(X_train, Y_train, 100, .001, 40, W2, b2, 0, X_val, Y_val)
    W3, b3 = miniBatchGD(X_train, Y_train, 100, .001, 40, W3, b3, .1, X_val, Y_val)
    W4, b4 = miniBatchGD(X_train, Y_train, 100, .001, 40, W4, b4, 1, X_val, Y_val)

    print("W1 train", computeAccuracy(X_train, y_train, W1, b1))
    print("W2 train", computeAccuracy(X_train, y_train, W2, b2))
    print("W3 train", computeAccuracy(X_train, y_train, W3, b3))
    print("W4 train", computeAccuracy(X_train, y_train, W4, b4))

    print("W1 test", computeAccuracy(X_test, y_test, W1, b1))
    print("W2 test", computeAccuracy(X_test, y_test, W2, b2))
    print("W3 test", computeAccuracy(X_test, y_test, W3, b3))
    print("W4 test", computeAccuracy(X_test, y_test, W4, b4))

    from functions import montage
    montage(W1)
    montage(W2)
    montage(W3)
    montage(W4)


if __name__ == "__main__":
    main()
