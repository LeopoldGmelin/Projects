import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels as sm
from numba import njit

X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])
L = X.shape[1]
n = []
n_1 = 2
for i in range(L):
    if i == 0:
        n.append(n_1)
    elif i == L - 1:
        n.append(1)
    else:
        n.append(n_1 + 1)
weights = [np.random.randn(n[i], n[i-1]) for i in range(1, L)]
biases = [np.random.randn(n[i], 1) for i in range(1, L)]

def prepare_data():
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    m = 10
    A_0 = X.T
    Y = y.reshape(n[-1], m)
    return A_0, Y

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_function_derivative(z):
    s = sigmoid_function(z)
    return s * (1 - s)

def cost(y_hat, y):
    # binary cross entropy loss function
    m = 10
    return (1/m) * (sum(-(y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat)))

@njit
def feed_forward_numba(A0, weights, biases, L):
    A = [None] * (L + 1)
    Z = [None] * (L + 1)
    A[0] = A0
    for i in range(1, L + 1):
        Z[i] = weights[i - 1] @ A[i - 1] + biases[i - 1]
        A[i] = sigmoid_function(Z[i])
    return A, Z

@njit
def backpropagation(A, Z, Y, L, weights, m):
    dcdZ = [None] * (L + 1)
    dcdW = [None] * (L + 1)
    dcdB = [None] * (L + 1)
    # Layer L: Fehler und Gradienten berechnen
    dcdZ[L] = A[L] - Y
    dcdW[L] = (1 / m) * (dcdZ[L] @ A[L - 1].T)
    dcdB[L] = (1 / m) * np.sum(dcdZ[L], axis=1, keepdims=True)
    # Für jede Schicht von L-1 bis 1
    for i in range(L - 1, 0, -1):
        dcdZ[i] = (weights[i].T @ dcdZ[i+1]) * sigmoid_function_derivative(Z[i])
        dcdW[i] = (1 / m) * (dcdZ[i] @ A[i - 1].T)
        dcdB[i] = (1 / m) * np.sum(dcdZ[i], axis=1, keepdims=True)
    return dcdW, dcdB, dcdZ


def train(alpha, A0, Y, L, weights, biases, epochs=10000):
    costs = np.zeros(epochs)
    m = Y.shape[1]
    for epoch in range(epochs):
        # Feedforward
        A, Z = feed_forward_numba(A0, weights, biases, L)
        y_hat = A[L]
        # Kostenberechnung
        costs[epoch] = (1/m) * np.sum(-Y * np.log(y_hat) - (1 - Y) * np.log(1 - y_hat))
        # Backpropagation
        dcdW, dcdB, dcdZ = backpropagation(A, Z, Y, L, weights, m)
        # Parameter-Update
        for i in range(1, L + 1):
            weights[i - 1] = weights[i - 1] - alpha * dcdW[i]
            biases[i - 1] = biases[i - 1] - alpha * dcdB[i]
    return weights, biases, costs

alpha = 0.01
epochs = 10000
A0, Y = prepare_data()
weights, biases, costs = train(alpha, A0, Y, L, weights, biases, epochs)
print("Training abgeschlossen. Letzter Kostenwert:", costs[-1])