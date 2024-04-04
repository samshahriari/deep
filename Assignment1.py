from functions import *
import numpy as np

NUM_CLASSES = 10
NUM_IMAGES = 10000
DIMENSION = 32*32*3  # 3072
np.random.seed(2424)


def loadBatchNP(batch_name: str):
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
    s = W @ X + b
    p = softmax(s)
    return p


def computeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b, lambda_) -> int:
    p = evaluateClassifier(X, W, b)
    J = np.sum(-Y.T @ np.log(p))/NUM_IMAGES + lambda_ * np.sum(W.flatten()**2)
    return J


def computeAccuracy(X, y, W, b):
    p = evaluateClassifier(X, W, b)
    prediction = p.argmax(axis=0)
    return np.sum(prediction == y) / len(y)


X, Y, y = loadBatchNP("data_batch_1")

X = preProcess(X)

# montage(X.T)

W = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
b = 0.01*np.random.randn(NUM_CLASSES, 1)


print(computeCost(X, Y, W, b, 1000))

print(computeAccuracy(X, y, W, b))
