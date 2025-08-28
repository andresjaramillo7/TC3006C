import os
import matplotlib.pyplot as plt
import numpy as np

# Obtenemos los datos y los cargamos
data = np.loadtxt("ex2data2.txt", delimiter=",")

# Creamos los vectores X: matriz (m,2) con las dos pruebas y Y: vector (m,) con 0s y 1s
X_vector = data[:, :2] # Matriz de características
y_vector = data[:, 2] # Vector de etiquetas

# Inicializamos los hiper parámetros
alpha = 0.01 # Iniciamos alpha/step en 0.01 para el gradiente descendente
iteraciones = 15000 # Número de epochs que hará el gradiente descendente
lambda_ = 1 # Parámetro de regularización

def graficaDatos(X, y, theta):
    """
    Grafica los datos de entrenamiento (x1 y x2) con 'x' si aceptado y 'o' si rechazado, y la frontera de decisión
    mapeada por theta, este es polinomial hasta el grado 6 en este caso.

    Parámetros:
        X: Matriz de características con datos originales sin mapear (m, 2).
        y: Vector de etiquetas (m,).
        theta: Vector de parámetros pesos previamente aprendidos.
    """
    # Definimos los puntos positivos y negativos
    positive = y == 1
    negative = y == 0

    # Graficamos los puntos aceptados ('x') y rechazados ('o')
    plt.scatter(X[positive, 0], X[positive, 1], marker='X', c='b', label='Aceptado')
    plt.scatter(X[negative, 0], X[negative, 1], marker='o', c='r', label='Rechazado')

    # Creamos una malla para graficar la frontera de decisión
    x_axis = np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200) # Rango de valores de x1
    y_axis = np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200) # Rango de valores de x2

    # Con meshgrid creamos todas las combinaciones posibles de x1 y x2
    X1, X2 = np.meshgrid(x_axis, y_axis)

    # Ahora usamos mesh_dots para obtener las predicciones y despues mapeamos x1 y x2 polinomialmente
    mesh_dots = np.c_[X1.ravel(), X2.ravel()]
    Z = mapeoCaracteristicas(mesh_dots) @ theta
    Z = Z.reshape(X1.shape) # Reshape para que coincida con la malla

    # Graficamos la frontera de decisión
    plt.contour(X1, X2, Z, levels=[0], linewidths=2, colors='g')

    plt.xlabel('Prueba 1'); plt.ylabel('Prueba 2')
    plt.legend(); plt.title('Datos y frontera de decisión')
    plt.show()

def mapeoCaracteristicas(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    degree = 6
    feats = [np.ones(x1.shape[0])] # bias

    for i in range(1, degree + 1):
        for j in range(i + 1):
            feats.append((x1 ** (i - j)) * (x2 ** j))

    return np.column_stack(feats)

X_mapped = mapeoCaracteristicas(X_vector)

def sigmoidal(z):
    return 1 / (1 + np.exp(-z))

def funcionCostoReg(theta, X, y, lambda_):
    m = y.size
    h = sigmoidal(X.dot(theta))

    J = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    gradient = (X.T @ (h - y)) / m
    gradient[1:] += (lambda_ / m) * theta[1:]
    
    return J, gradient

lambda_ = 1

theta0 = np.zeros(X_mapped.shape[1])
J0, g0 = funcionCostoReg(theta0, X_mapped, y_vector, lambda_)
print("J(theta=0, λ=1) =", J0)

def aprende(theta, X, y, iteraciones):
    lambda_ = 1
    for epoch in range(iteraciones):
        J, grad = funcionCostoReg(theta, X, y, lambda_)
        theta -= alpha * grad
    return theta

def predice(X, theta):
    prob = sigmoidal(X.dot(theta))
    return (prob >= 0.5).astype(int)

theta = aprende(theta0, X_mapped, y_vector, iteraciones)
predicciones = predice(X_mapped, theta)
print("Predicciones:", predicciones)
accuracy = np.mean(predicciones == y_vector) * 100
print(f"Accuracy: {round(accuracy, 6)}%")
"""
Llamamos a las funciones para obtener primero el costo sin modificar las thetas,
después las ajustamos con la gradiente descendente y graficamos los datos.
"""
graficaDatos(X_vector, y_vector, theta)
# print(calculaCosto(X1_list, X2_list, y_list, theta)) # Cálculo del costo inicial
# gradienteDescendente(X1_list, X2_list, y_list, theta, alpha, iteraciones) # Ajuste de thetas
# graficaDatos(X1_list, X2_list, y_list, theta) # Gráfica de los datos