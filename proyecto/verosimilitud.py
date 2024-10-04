import autograd.numpy as np
from autograd import *
import pandas as pd
# Función sigmoide
#mapea cualquier valor real a un valor entre 0 y 1
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Evitar desbordamiento en exp
    return 1 / (1 + np.exp(-z))

# Función de hipótesis
#calcula la probabilidad de que una observación pertenezca a la clase positiva 
def hypothesis(w, x):
    return sigmoid(np.dot(w, x))

# Función de log-verosimilitud
# verosimilitud es la probabilidad de observar los datos dados ciertos parámetros del modelo
# método para estimar los parámetros de un modelo estadístico. El objetivo es encontrar los valores de que maximicen la función de verosimilitud
def log_likelihood(w, X, y):
    m = X.shape[0]  # Número de filas de la matriz X
    total_likelihood = 0
    epsilon = 1e-15  # Pequeña constante para estabilizar el logaritmo y evitar log(0)
    for i in range(m):  # Itera sobre cada muestra
        h = hypothesis(w, X[i])  # Calcula la hipótesis para la muestra i
        h = np.clip(h, epsilon, 1 - epsilon)  # Asegura que h nunca sea exactamente 0 o 1
        total_likelihood += y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h)  # Suma la log-verosimilitud de la muestra i
    return total_likelihood
