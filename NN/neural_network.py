import numpy as np

class NN:
    def __init__(self):
        self.learning_rate = 0.1
        self.model = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_deriv(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def LeakyReLU(x):
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))  # evitar overflow numérico
        return exps / np.sum(exps)

    @staticmethod
    def cross_entropy_loss(y, y_hat):
        epsilon = 1e-10
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(y_hat))

    @staticmethod
    def cross_entropy_loss_deriv(y, y_hat):
        return y_hat - y
    
    @staticmethod
    def foward_pass(x, W, bias):
        hs = []
        zs = []
        h = np.array(x).reshape(-1, 1)  # Redimensionar entrada
        hs.append(h)
        for l in range(len(W) - 1):
            z = W[l] @ h + bias[l]
            zs.append(z)
            h = NN.ReLU(z)
            hs.append(h)

        z = W[-1] @ h + bias[-1]
        zs.append(z)
        y = NN.sigmoid(z)  # usar sigmoid na saída para classificação binária
        hs.append(y)
        return (hs, zs)

    @staticmethod
    def backward_pass(W, b, y, hs, zs, learning_rate):
        d_act_z = NN.sigmoid(hs[-1]) * (1 - NN.sigmoid(hs[-1]))  # derivada da sigmoid
        dL_dy = NN.cross_entropy_loss_deriv(y, hs[-1])
        dL_dz = dL_dy * d_act_z

        dL_dWs = [dL_dz @ hs[-2].T]
        dL_dbs = [dL_dz]

        grad_loss_l = dL_dz

        for idx_l in range(len(W) - 2, -1, -1):
            grad_loss_l = np.dot(W[idx_l + 1].T, grad_loss_l) * NN.ReLU_deriv(zs[idx_l])
            dL_dWs.insert(0, grad_loss_l @ hs[idx_l].T)
            dL_dbs.insert(0, grad_loss_l)

        for i in range(len(W)):
            W[i] -= learning_rate * dL_dWs[i]
            b[i] -= learning_rate * dL_dbs[i]
        return W, b

    def fit(self, X, y):
        np.random.seed(42)  # Definir seed para reprodutibilidade
        W = [np.random.rand(3, 3), np.random.rand(3, 3), np.random.rand(1, 3)]
        b = [np.random.rand(3, 1), np.random.rand(3, 1), np.random.rand(1, 1)]
        epochs = 1000

        for t in range(epochs):
            for x_idx in range(len(X)):
                # Forward Pass
                hs, zs = NN.foward_pass(X[x_idx], W, b)

                # Backward Pass
                W, b = NN.backward_pass(W, b, y[x_idx], hs, zs, self.learning_rate)
                
        self.model = (W, b)

    def predict(self, X):
        y = np.zeros(len(X))
        for x_idx in range(len(X)):
            hs, zs = NN.foward_pass(X[x_idx], self.model[0], self.model[1])
            y[x_idx] = 1 if hs[-1] > 0.5 else 0  # Classificação binária
        return y