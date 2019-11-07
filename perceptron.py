import numpy as np

TRAINING_ITERATIONS = 20000
SEED = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(SEED)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random weights:\n", synaptic_weights)

for iterations in range(TRAINING_ITERATIONS):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs
    weight_deltas = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, weight_deltas)

print("Weights after:\n", synaptic_weights)
print("Outputs:\n", outputs)
