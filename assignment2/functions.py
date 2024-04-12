import copy
import numpy as np
from Assignment2 import computeCost


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('../Datasets/cifar-10-batches-py/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ComputeGradsNum(X, Y, P, W, b, lambda_, h):
    #     W1 = W[0]
    #     W2 = W[1]
    #     b1 = b[0]
    #     b2 = b[1]
    #     grad_W2 = np.zeros(shape=W2.shape)
    #     grad_b2 = np.zeros(shape=b2.shape)
    #     grad_W1 = np.zeros(shape=W1.shape)
    #     grad_b1 = np.zeros(shape=b1.shape)
    grad_W = [np.zeros(W_i.shape) for W_i in W]
    grad_b = [np.zeros(b_i.shape) for b_i in b]

    c = computeCost(X, Y, W, b, lambda_)

    for l in range(len(b)):
        for i in range(len(b[l])):
            b_try = copy.deepcopy(b)
            b_try[l][i] += h
            c2 = computeCost(X, Y, W, b_try, lambda_)
            grad_b[l][i] = (c2-c)/h

    for l in range(len(W)):
        for i in range(W[l].shape[0]):
            for j in range(W[l].shape[1]):
                W_try = copy.deepcopy(W)
                W_try[l][i, j] += h
                c2 = computeCost(X, Y, W_try, b, lambda_)
                grad_W[l][i, j] = (c2-c) / h

    return grad_W, grad_b


def ComputeGradsNumSlow(X, Y, P, W, b, lambda_, h):
    grad_W = [np.zeros(W_i.shape) for W_i in W]
    grad_b = [np.zeros(b_i.shape) for b_i in b]

    for l in range(len(b)):
        for i in range(len(b[l])):
            b_try = [b_i.copy() for b_i in b]
            b_try[l][i] -= h
            c1 = computeCost(X, Y, W, b_try, lambda_)

            b_try = [b_i.copy() for b_i in b]
            b_try[l][i] += h
            c2 = computeCost(X, Y, W, b_try, lambda_)

            grad_b[l][i] = (c2 - c1) / (2 * h)

    for l in range(len(W)):
        for i in range(W[l].shape[0]):
            for j in range(W[l].shape[1]):
                W_try = copy.deepcopy(W)
                W_try[l][i, j] -= h
                c1 = computeCost(X, Y, W_try, b, lambda_)

                W_try = copy.deepcopy(W)
                W_try[l][i, j] += h
                c2 = computeCost(X, Y, W_try, b, lambda_)

                grad_W[l][i, j] = (c2-c1) / (2*h)

    return grad_W, grad_b


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 10)
    for i in range(2):
        for j in range(5):
            im = W[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i*5+j].imshow(sim, interpolation='nearest')
            # ax[i][j].set_title("y="+str(5*i+j))
            ax[i*5+j].axis('off')
    plt.show()

# def save_as_mat(data, name="model"):
# 	""" Used to transfer a python model to matlab """
# 	import scipy.io as sio
# 	sio.savemat(name'.mat',{name:b})
