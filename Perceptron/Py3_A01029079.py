import os
import numpy as np

# Obtenemos los datos y los cargamos en este caso son 5000 muestras de:
# - 400 features (pixeles en una imágen 20x20)
# - 1 label que dice si es un digito del 1 al 10
data = np.loadtxt("digitos.txt")

# Matriz de features de X (m, 400)
X = data[:, :400].astype(np.float64)
# Vector de etiquetas y (m,)
y = data[:, 400].astype(np.int64)

# Funciones auxiliares
def sigmoid(z):
    """
    Ejecuta la función sigmoidal elemento a elemento. Fórmula: g(z) = 1 / (1 + e^(-z))

    Parámetros:
        z: valores de la red neuronal

    Regresa:
        g(z) con la misma forma que z
    """
    return 1 / (1 + np.exp(-z))

def randInicializacionPesos(L_in, L_out):
    """
    Inicializa aleatoriamente una matriz de pesos en el rango [-0.12, 0.12]. Se usa para romper la simetría entre neuronas.

    Parámetros:
        L_in:  número de features hacia la capa actual
        L_out: número de neuronas en la capa actual

    Regresa:
        W: matriz de pesos de forma (L_out, L_in)
           (no incluye el bias; el bias se maneja por separado en b1/b2)
    """
    return np.random.rand(L_out, L_in) * 2 * 0.12 - 0.12

def sigmoidalGradiente(z):
    """
    Calcula la derivada de la función sigmoidal. Fórmula: g'(z) = g(z) * (1 - g(z)), donde g es la sigmoide.

    Parámetros:
        z: valores de la red neuronal

    Regresa:
        derivada sigmoidal con la misma forma que z
    """
    return sigmoid(z) * (1 - sigmoid(z))

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    """
    Entrena una red neuronal con una sola capa oculta usando gradiente descendiente (batch),
    activación sigmoidal y one-vs-all para la salida multicapa.

    Arquitectura:
        Entrada (input_layer_size=400)
            -> Oculta (hidden_layer_size=25)
            -> Salida (num_labels=10)

    Parámetros:
        input_layer_size  : número de unidades de entrada (400)
        hidden_layer_size : número de neuronas en la capa oculta (25)
        num_labels        : número de clases (10)
        X                 : matriz (m, 400)
        y                 : vector (m,)

    Regresa:
        W1 : pesos capa oculta de forma (25, 400)
        b1 : bias capa oculta de forma (25,)
        W2 : pesos capa salida de forma (10, 25)
        b2 : bias capa salida de forma (10,)
        J  : último valor del costo promedio
    """
    np.random.seed(7) # Seed para replicar la reproducibilidad

    # Inicialización de los hiperparámetros
    alpha = 0.9 # Iniciamos alpha/step en 0.9
    epochs = 1500 # Número de epochs
    m, n = X.shape # m = #ejemplos, n = #características (=400)

    # Hacemos one-hot encoding de las etiquetas
    # (10 = dígito '0' se mapea a la columna 9)
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    # Inicialización de los pesos y bias (los bias son ceros por su estándar para sigmoide)
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    b1 = np.zeros(hidden_layer_size)
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)
    b2 = np.zeros(num_labels)

    J = 0 # Inicializamos el costo fuera para que se reinicie en cada epoch

    for epoch in range(epochs):
        # Feedforward
        # Net1/O1: capa oculta
        Net1 = X.dot(W1.T) + b1
        O1 = sigmoid(Net1)
        # Net2/O2: capa de salida
        Net2 = O1.dot(W2.T) + b2
        O2 = sigmoid(Net2)

        # Cálculo de la función de costo
        J = -np.sum(Y * np.log(O2) + (1 - Y) * np.log(1 - O2)) / m

        # Backpropagation
        # Error en salida (capa 2): derivada del costo respecto a Net2
        error2 = O2 - Y
        # Gradientes para W2 y b2
        pepW2 = (error2.T.dot(O1)) / m
        pepb2 = np.sum(error2, axis=0) / m

        # Error en oculta (capa 1): retropropagamos error2 por W2 y multiplicamos por g'(Net1)
        error1 = error2.dot(W2) * sigmoidalGradiente(Net1)
        # Gradientes para W1 y b1
        pepW1 = (error1.T.dot(X)) / m
        pepb1 = np.sum(error1, axis=0) / m

        # Actualización de los pesos
        W2 -= alpha * pepW2
        b2 -= alpha * pepb2
        W1 -= alpha * pepW1
        b1 -= alpha * pepb1

        # Log de progreso cada 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Costo: {round(J, 6)}")

    return W1, b1, W2, b2, J

def prediceRNYaEntrenada(X, W1, b1, W2, b2):
    """
    Realiza el feedforward con los parámetros entrenados y devuelve la clase predicha para cada ejemplo.

    Parámetros:
        X  : matriz (m, 400) de features
        W1 : (25, 400) pesos de capa oculta
        b1 : (25,)     bias de capa oculta
        W2 : (10, 25)  pesos de capa salida
        b2 : (10,)     bias de capa salida

    Regresa:
        prediccion: vector (m,) con valores del 1 al 10 por el one-hot encoding
    """
    # Feedforward
    Net1 = X.dot(W1.T) + b1
    O1 = sigmoid(Net1)
    Net2 = O1.dot(W2.T) + b2
    O2 = sigmoid(Net2)
    # np.argmax devuelve los resultados entre 0 y 9, el +1 los vuelve del 1 al 10
    prediccion = np.argmax(O2, axis=1) + 1
    return prediccion

"""
Por último se llaman a las funciones, primero se entrenan los pesos y se obtiene el costo,
después se realizan las predicciones y se obtiene el accuracy, realizando un print final.
"""
W1, b1, W2, b2, J = entrenaRN(400, 25, 10, X, y)
predicciones = prediceRNYaEntrenada(X, W1, b1, W2, b2)
acc = np.mean(predicciones == y) * 100
print(f"Precisión en el conjunto de entrenamiento: {round(acc, 2)}%")