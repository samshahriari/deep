from functions import *
import numpy as np

NUM_CLASSES = 10
NUM_IMAGES = 10000
DIMENSION = 32*32*3


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


def preProcess(X):
    X -= np.mean(X, axis=1, keepdims=True)
    X /= np.std(X, axis=1, keepdims=True)
    return X


def evaluateClassifier(X, W, b):
    pass


X, Y, y = loadBatchNP("data_batch_1")

X = preProcess(X)

montage(X.T)

W = 0.01*np.random.randn(NUM_CLASSES, DIMENSION)
b = 0.01*np.random.randn(NUM_CLASSES, 1)
