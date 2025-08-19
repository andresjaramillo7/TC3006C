import numpy as np

training_data = np.array([
    [1, 0.8, 0.2, 0.9, 0.7, 0.5, 1],
    [1, -0.5, 0.7, 0.3, 0.2, 1.1, 0],
    [1, 0.9, 0.1, 0.8, 0.9, 0.6, 1],
    [1, 0.0, 0.5, 0.4, 0.3, 0.8, 0],
    [1, 0.7, 0.3, 0.7, 0.8, 0.4, 1],
    [1, -0.1, 0.8, 0.1, 0.1, 1.5, 0],
    [1, 0.6, 0.1, 0.6, 0.7, 0.5, 1],
    [1, 0.2, 0.4, 0.2, 0.4, 0.9, 0],
    [1, 1.0, 0.0, 1.0, 1.0, 0.3, 1],
    [1, -0.2, 0.6, 0.0, 0.0, 1.2, 0],
    [1, 0.5, 0.2, 0.7, 0.6, 0.3, 1],
    [1, 0.1, 0.3, 0.4, 0.5, 0.7, 0]
])

test_data = np.array([
    [1, 0.9, 0.0, 0.9, 0.9, 0.6, 1],
    [1, -0.3, 0.5, 0.1, 0.2, 1.0, 0],
    [1, 0.8, 0.1, 0.8, 0.8, 0.4, 1]
])

weights = np.random.rand(6,1)

step = 0.01

def activation_function(y):
    return 1 if y >= 0 else 0

def output_function(sample, weights):
    x = sample[:-1]
    y = np.dot(x, weights)
    return activation_function(y)

def error(y, y_hat):
    return y - y_hat

def update_weights(weights, sample, error, step):
    x = sample[:-1].reshape(-1, 1)
    return weights + step * error * x

def train(training_data, weights, step, max_epochs=100):
    for epoch in range(max_epochs):
        epoch_errors = 0
        print(f"\nEpoch {epoch+1}") 
        for i, sample in enumerate(training_data, 1):
            y = sample[-1]
            y_hat = output_function(sample, weights)
            err = error(y, y_hat)
            if err != 0:
                weights = update_weights(weights, sample, err, step)
                epoch_errors += 1
            print(f" Sample {i}: y_real={y}, y_hat={y_hat}, error={err}")
        print(f" Total errors in epoch {epoch+1}: {epoch_errors}")
        if epoch_errors == 0:
            print(f"\nEnded at epoch {epoch+1}")
            break
    return weights

final_weights = train(training_data, weights, step, max_epochs=100)

print("\nFinal weights after training:")
print(final_weights)