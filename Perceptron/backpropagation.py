import numpy as np

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

alfa = 0.1

b3 = np.random.rand(1)
b4 = np.random.rand(1)
b5 = np.random.rand(1)
w13 = np.random.rand(1)
w23 = np.random.rand(1)
w14 = np.random.rand(1)
w24 = np.random.rand(1)
w35 = np.random.rand(1)
w45 = np.random.rand(1)

for epoch in range(20000):
    error_epoch = 0
    for i in range(4):
        x1 = dataset_xor[i][1]
        x2 = dataset_xor[i][2]
        y = dataset_xor[i][3]

        print("Entradas:", x1, x2, "Salida esperada:", y)

        Net3 = b3 + w13 * x1 + w23 * x2
        Net4 = b4 + w14 * x1 + w24 * x2

        O3 = 1 / (1 + np.exp(-Net3))
        O4 = 1 / (1 + np.exp(-Net4))

        Net5 = b5 + w35 * O3 + w45 * O4
        O5 = 1 / (1 + np.exp(-Net5))

        pepw35 = -(y - O5) * O5 * (1 - O5) * O3
        pepw45 = -(y - O5) * O5 * (1 - O5) * O4
        pepb5 = -(y - O5) * O5 * (1 - O5)
        pepw13 = -(y - O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3) * x1
        pepw23 = -(y - O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3) * x2
        pepb3 = -(y - O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3)
        pepw14 = -(y - O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4) * x1
        pepw24 = -(y - O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4) * x2
        pepb4 = -(y - O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4)

        w35 = w35 - alfa * pepw35
        w45 = w45 - alfa * pepw45
        b5 = b5 - alfa * pepb5
        w13 = w13 - alfa * pepw13
        w23 = w23 - alfa * pepw23
        b3 = b3 - alfa * pepb3
        w14 = w14 - alfa * pepw14
        w24 = w24 - alfa * pepw24
        b4 = b4 - alfa * pepb4

        e = (1/2) * (y - O5) ** 2
        error_epoch += e[0]
        print("O5:", O5)
    print("Error:", round(error_epoch, 4), "%", "Epoch:", epoch)
    if (error_epoch < 0.001):
        break