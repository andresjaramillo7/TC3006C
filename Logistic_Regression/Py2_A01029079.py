import os
import matplotlib.pyplot as plt
import numpy as np

# Obtener ruta absoluta y ruta al .txt con los datos
base = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(base, 'ex2data1.txt')

# Se crean las listas para guardar los valores de "X" (en este caso son de población) y de "y" (que son las ganancias)
X1_list = []
X2_list = []
y_list = []

# Leemos el archivo omitiendo el encabezado y guardando los valores en las listas
with open(data, 'r', encoding = "latin-1") as f:
    for line in f:
        values = line.split(",")
        X1_list.append(float(values[0]))
        X2_list.append(float(values[1]))
        y_list.append(float(values[2]))
    f.close()

print(X1_list)
print(X2_list)
print(y_list)

# Inicializamos los parámetros
theta = [0.0, 0.0, 0.0] # Inicializamos theta 0, theta 1 y theta 2 en 0s
alpha = 0.001 # Iniciamos alpha/step en 0.01 como se indica
iteraciones = 10000 # Número de epochs que hará el gradiente descendente

def z(theta, xi1, xi2):
    """Calcula la z de la regresión logística. Fórmula: z = theta0 + theta1 * xi1 + theta2 * xi2"""
    return theta[0] + theta[1] * xi1 + theta[2] * xi2

def hipothesis(theta, xi1, xi2):
    """Calcula la hipótesis de la regresión logaritmica. Fórmula: h(x) = 1 / (1 + exp(-(theta0 + theta1 * xi1 + theta2 * xi2)))"""
    return 1 / (1 + np.exp(-z(theta, xi1, xi2)))

def error(theta, xi1, xi2, yi):
    """Calcula el error de la regresión logarítmica de un solo punto. Fórmula: yi * log(h(x)) + (1 - yi) * log(1 - h(x))"""
    return yi * np.log(hipothesis(theta, xi1, xi2)) + (1 - yi) * np.log(1 - hipothesis(theta, xi1, xi2))

def graficaDatos(X, y, theta):
    """
    Grafica los datos de entrenamiento (en este caso X_list y y_list) y la recta de regresión lineal
    después de obtener los mejores valores de theta.
    
    Parámetros:
        X: Es una lista de valores de población (X_list).
        y: Es una lista de valores de ganancias (y_list).
        theta: Es una lista de parámetros de la regresión (theta0, theta1).
    """
    plt.scatter(X, y, color='red') # Datos de entrenamiento
    
    x_min, x_max = min(X), max(X) # Obtienes los valores mínimos y máximos de x
    y_min = hipothesis(theta, x_min) # Obtienes el valor mínimo de y
    y_max = hipothesis(theta, x_max) # Obtienes el valor máximo de y
    plt.plot([x_min, x_max], [y_min, y_max], color='blue') # Recta de regresión

    plt.legend(["Datos de entrenamiento", "Recta de la regresión"])
    plt.xlabel("Población (x)")
    plt.ylabel("Ganancias (y)")
    plt.title("Proyecto 1 - Regresión Lineal Simple")
    plt.show()

def gradienteDescendente(X1, X2, y, theta, alpha, iteraciones):
    """
    Realiza el algoritmo de gradiente descendente para encontrar los mejores parámetros theta,
    utilizando el método "batch" (iterar sobre todos los ejemplos).
    
    Parámetros:
    X: Lista de valores de población (X_list).
    y: Lista de valores de ganancias (y_list).
    theta: Lista de parámetros de la regresión (theta0, theta1).
    alpha: Tasa/Paso de aprendizaje.
    iteraciones: Número de iteraciones de época para el gradiente descendente.
    """
    m = len(y)
    for epoch in range(iteraciones):
        temp0 = 0.0
        temp1 = 0.0
        temp2 = 0.0
        for i in range(m):
            e = hipothesis(theta, X1[i], X2[i]) - y[i]
            temp0 += e
            temp1 += e * X1[i]
            temp2 += e * X2[i]
        theta[0] -= (alpha / m) * temp0
        theta[1] -= (alpha / m) * temp1
        theta[2] -= (alpha / m) * temp2
        print("Costo en iteracion", epoch, "-->", calculaCosto(X1, X2, y, theta)) # Imprime el costo en cada iteración
    print("Theta final:", theta) # Imprime los valores finales de theta
    return theta

def calculaCosto(X1, X2, y, theta):
    """
    Calcula la funciòn de costo para la regresiòn lineal.

    Parámetros:
        X: Lista de valores de población (X_list).
        y: Lista de valores de ganancias (y_list).
        theta: Lista de parámetros de la regresión (theta0, theta1).
    """
    m = len(y)
    sum = 0.0
    for i in range(m):
        e = error(theta, X1[i], X2[i], y[i])
        sum += e
    return -sum / m
"""
Llamamos a las funciones para obtener primero el costo sin modificar las thetas,
después las ajustamos con la gradiente descendente y graficamos los datos.
"""
print(calculaCosto(X1_list, X2_list, y_list, theta)) # Cálculo del costo inicial
gradienteDescendente(X1_list, X2_list, y_list, theta, alpha, iteraciones) # Ajuste de thetas
# graficaDatos(X1_list, X2_list, y_list, theta) # Gráfica de los datos