# from functions import *
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


# vi vill minimera kostnaden för att få så bra parametrar som möjligt
def computeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_) -> float:
    p = evaluateClassifier(X, W, b)
    J = np.sum(-Y * np.log(p))/X.shape[1] + lambda_ * np.sum(W**2)  # want to multiply element wise
    return J


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


def main():
    X, Y, y = loadBatchNP("data_batch_1")

    X = preProcess(X)

    # montage(X.T)

    W = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
    b = 0.01*np.random.randn(NUM_CLASSES, 1)

    # print(X.shape)

    # print(computeCost(X, Y, W, b, 1000))

    # print(computeAccuracy(X, y, W, b))

    evaluateGradient(X, Y, W, b)


if __name__ == "__main__":
    main()
