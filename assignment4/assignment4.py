import numpy as np


class RNN:
    def __init__(self, file_path) -> None:
        np.random.seed(2424)

        self.book_data = []
        self.i2c = []
        self.c2i = {}

        with open(file_path, "r", encoding="utf-8") as f:
            self.book_data = list(f.read())
        for c in self.book_data:
            if c not in self.c2i:
                self.c2i[c] = len(self.i2c)
                self.i2c.append(c)
        self.K = len(self.i2c)
        self.m = 5
        self.eta = .1
        self.seq_length = 25
        self.RNN = {}
        self.RNN["b"] = np.zeros((self.m))
        self.RNN["c"] = np.zeros((self.K))
        self.sig = .01
        self.RNN["U"] = np.random.randn(self.m, self.K) * self.sig
        self.RNN["W"] = np.random.randn(self.m, self.m) * self.sig
        self.RNN["V"] = np.random.randn(self.K, self.m) * self.sig

        X_chars = self.book_data[:self.seq_length]
        Y_chars = self.book_data[1:self.seq_length+1]
        self.X = np.zeros((self.K, self.seq_length))
        for i, j in enumerate(X_chars):
            self.X[self.c2i[j], i] = 1
        self.Y = np.zeros((self.K, self.seq_length))
        for i, j in enumerate(Y_chars):
            self.Y[self.c2i[j], i] = 1
        self.h0 = np.zeros((self.m))

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def onehot(self, c=None, i=None):
        print(i)
        if c is not None:
            x = np.zeros((self.K))
            x[self.c2i[c]] = 1
            return x
        else:
            x = np.zeros((self.K))
            x[i] = 1
            return x

    def synthesize(self, n, x0=None):
        if x0 == None:
            x0 = self.onehot(c=".")
        h = [self.h0]
        x = [x0]
        a = np.zeros((n, self.m))
        o = np.zeros((n, self.K))
        p = np.zeros((n, self.K))
        # Y = np.zeros((self.K, n))
        Y = []
        for t in range(n):
            # print(t)
            # print(x[-1])
            # print((self.RNN["W"] @ h[-1]))
            # print((self.RNN["U"] @ x[-1]))
            # print(self.RNN["b"])
            # print((self.RNN["W"] @ h[-1]) + (self.RNN["U"] @ x[-1]) + self.RNN["b"])
            # print("a", a)
            a[t] = ((self.RNN["W"] @ h[-1]) + (self.RNN["U"] @ x[-1]) + self.RNN["b"])
            h.append(np.tanh(a[t]))
            o[t] = self.RNN["V"]@h[-1] + self.RNN["c"]
            p[t] = self.softmax(o[t])
            cp = np.cumsum(p[t])

            ixs = np.where(cp - np.random.rand() > 0)
            ii = ixs[0][0]
            x.append(self.onehot(i=ii))
            Y.append(ii)
        return Y

    def forward_pass(self, X, Y, RNN=None):
        if RNN is None:
            RNN = self.RNN
        h = [self.h0]
        a = np.zeros((X.shape[1], self.m))
        o = np.zeros((X.shape[1], self.K))
        p = np.zeros((X.shape[1], self.K))
        for t in range(X.shape[1]):
            a[t] = ((RNN["W"] @ h[-1]) + (RNN["U"] @ X[:, t]) + RNN["b"])
            h.append(np.tanh(a[t]))
            print((self.RNN["V"]@h[-1]).shape)
            print(self.RNN["c"].shape)
            o[t] = RNN["V"]@h[-1] + RNN["c"]
            p[t] = self.softmax(o[t])

        return -np.sum(Y.T*np.log(p)), o, h[1:], p

    def backward_pass(self, X, Y, p, h):
        self.grads = {}
        self.grads["V"] = np.zeros_like(self.RNN["V"])
        self.grads["c"] = np.zeros_like(self.RNN["c"])
        self.grads["W"] = np.zeros_like(self.RNN["W"])
        self.grads["U"] = np.zeros_like(self.RNN["U"])
        self.grads["b"] = np.zeros_like(self.RNN["b"])
        for t in range(X.shape[1]-1, -1, -1):
            grad_o_t = -(Y[:, t]-p[t])
            self.grads["V"] += np.outer(grad_o_t, h[t])
            self.grads["c"] += grad_o_t.T
            if t == X.shape[1]-1:  # t = tau
                grad_h_t = grad_o_t @ self.RNN["V"]
            else:
                grad_h_t = grad_o_t @ self.RNN["V"] + grad_a_t @ self.RNN["W"]
            grad_a_t = grad_h_t @ np.diag(1 - h[t]**2)
            if t == 0:
                self.grads["W"] += np.outer(grad_a_t, self.h0)
            else:
                self.grads["W"] += np.outer(grad_a_t, h[t-1])
            self.grads["U"] += np.outer(grad_a_t, X[:, t])
            self.grads["b"] += grad_a_t
        for f in self.grads.keys():
            self.grads[f] = np.clip(self.grads[f], -5, 5)

    def compare_grads(self):
        _, _, h, p = self.forward_pass(self.X, self.Y)
        self.backward_pass(self.X, self.Y, p, h)
        for f in self.RNN.keys():
            num_grad = self.ComputeGradNum(f, 1e-4)
            diff = num_grad-self.grads[f]
            print(f, "max : ", np.max(np.abs(diff)))

    def ComputeLoss(self, RNN=None):
        return self.forward_pass(self.X, self.Y, RNN)[0]

    def ComputeGradNum(self, f, h):
        import copy
        grad = np.zeros_like(self.RNN[f])
        if f in ["b", "c"]:
            for i in range(self.RNN[f].shape[0]):
                RNN_try = copy.deepcopy(self.RNN)
                RNN_try[f][i] -= h
                l1 = self.ComputeLoss(RNN_try)

                RNN_try = copy.deepcopy(self.RNN)
                RNN_try[f][i] += h
                l2 = self.ComputeLoss(RNN_try)

                grad[i] = (l2 - l1) / (2 * h)

            return grad
        for i in range(self.RNN[f].shape[0]):
            for j in range(self.RNN[f].shape[1]):
                RNN_try1 = copy.deepcopy(self.RNN)
                RNN_try1[f][i, j] -= h
                # print(RNN_try)
                l1 = self.ComputeLoss(RNN_try1)

                RNN_try2 = copy.deepcopy(self.RNN)
                RNN_try2[f][i, j] += h
                l2 = self.ComputeLoss(RNN_try2)
                # print(RNN_try1[f]-RNN_try2[f])

                grad[i, j] = (l2 - l1) / (2 * h)

        return grad

    def train(self):
        pass


if __name__ == "__main__":
    r = RNN("goblet_book.txt")
    # r.compare_grads()
    story = r.synthesize(100)
    for c in story:
        print(r.i2c[c], end="")
