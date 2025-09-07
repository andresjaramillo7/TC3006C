import numpy as np
import matplotlib.pyplot as plt

dataset_and = np.array([[1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0]])

dataset_or = np.array([[1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0]])

dataset_xor = np.array([[1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 0.0]])

w0 = 1.5
w1 = 0.5
w2 = 1.5

alfa = 0.1

def activation_function(y):
    return 1 if y >= 0 else 0

for epoch in range(1000):
    print("Época:", epoch)
    for i in range(len(dataset_or)):
        x0 = dataset_or[i][0]
        x1 = dataset_or[i][1]
        x2 = dataset_or[i][2]
        d = dataset_or[i][3]
        Netk = w0 * x0 + w1 * x1 + w2 * x2
        output = activation_function(Netk)
        error = d - Netk
        w0 += alfa * error * x0
        w1 += alfa * error * x1
        w2 += alfa * error * x2
        print("w0:", w0, "w1:", w1, "w2:", w2)

print("Pesos finales:")
print("w0:", w0)
print("w1:", w1)
print("w2:", w2)
plt.show()