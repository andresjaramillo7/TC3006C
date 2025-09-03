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
    Z = mapeoCaracteristicas(mesh_dots) @ theta # Predicciones
    Z = Z.reshape(X1.shape) # Reshape para que coincida con la malla

    # Graficamos la frontera de decisión
    plt.contour(X1, X2, Z, levels=[0], linewidths=2, colors='g')

    plt.xlabel('Prueba 1'); plt.ylabel('Prueba 2')
    plt.legend(); plt.title('Datos y frontera de decisión')
    plt.show()

def mapeoCaracteristicas(X):
    """
    Mapea las características polinomialmente hasta el grado, también agregando el bias.
    Siguiendo la fórmula: x1^i-j * x2^j
    Retorna: Matriz de características mapeadas (m, 28) al tener grado 6.

    Parámetros:
        X: Matriz de características originales (m, 2).
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    degree = 6
    feats = [np.ones(x1.shape[0])] # bias

    # Genera x1^(i-j) * x2^j para todo 1<=i<=degree y 0<=j<=i
    for i in range(1, degree + 1):
        for j in range(i + 1):
            feats.append((x1 ** (i - j)) * (x2 ** j))

    return np.column_stack(feats)

def sigmoidal(z):
    """
    Ejecuta la función sigmoidal. Fórmula: 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))

def funcionCostoReg(theta, X, y, lambda_):
    """
    Calcula el costo de la regresión logística con regularización.
    También calcula el gradiente descendente ya vectorizado.
    
    Parámetros:
        theta: Vector de parámetros pesos (n,).
        X: Matriz de características (m, n).
        y: Vector de etiquetas (m,).
        lambda_: Parámetro de regularización.
    """
    m = y.size # Número de ejemplos
    h = sigmoidal(X.dot(theta)) # Hipótesis

    # Función de costo para regresión lineal
    J = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    
    gradient = (X.T @ (h - y)) / m # Gradiente cuando es j = 0
    gradient[1:] += (lambda_ / m) * theta[1:] # Gradiente cuando j >= 1
    
    return J, gradient

def aprende(theta, X, y, iteraciones):
    """
    Entrenamiento con la gradiente descendiente de funcionCostoReg.
    También actualiza theta en cada época con: theta -= alpha * grad
    
    Parámetros:
        theta: Vector de parámetros pesos (n,).
        X: Matriz de características (m, n).
        y: Vector de etiquetas (m,).
        iteraciones: Número de épocas para el entrenamiento.
    """
    for epoch in range(iteraciones):
        J, grad = funcionCostoReg(theta, X, y, lambda_)
        theta -= alpha * grad # Actualiza theta
    return theta

def predice(X, theta):
    """
    Realiza predicciones utilizando el modelo entrenado con un umbral de 0.5.
    Retorna un vector de predicciones con 1s y 0s.
    
    Parámetros:
        X: Matriz de características (m, n).
        theta: Vector de parámetros pesos ya entrenado (n,).
    """
    prob = sigmoidal(X.dot(theta))
    return (prob >= 0.5).astype(int)

"""
Llamamos a las funciones, primero mapeamos el vector de las X para que sea de grado 6.
Después creamos nuestro vector de parámetros iniciales theta0 con el tamaño de la matriz de características mapeadas.
Luego ajustamos las thetas utilizando el algoritmo de gradiente descendente.
Al mismo tiempo entrenamos el modelo y después realizamos las predicciones.
Hacemos un print de las predicciones y la precisión.
Terminamos graficando los datos y la frontera de decisión.
"""
X_mapped = mapeoCaracteristicas(X_vector)
theta0 = np.zeros(X_mapped.shape[1])
theta = aprende(theta0, X_mapped, y_vector, iteraciones)
predicciones = predice(X_mapped, theta)
print("Predicciones:", predicciones)
accuracy = np.mean(predicciones == y_vector) * 100
print(f"Accuracy: {round(accuracy, 6)}%")
graficaDatos(X_vector, y_vector, theta)