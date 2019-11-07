import numpy as np

TRAINING_ITERATIONS = 100000
SEED = 1
np.set_printoptions(formatter={'all':lambda x: str(x)})

class NN:

    def __init__(self, seed=SEED):
        np.random.seed(seed)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return self. __sigmoid(x) * (1 - self. __sigmoid(x))

    def train(self, t_inputs, t_outputs, iters=TRAINING_ITERATIONS):
        for iterations in range(iters):
            output = self.think(t_inputs)
            error = t_outputs - output
            self.synaptic_weights += np.dot(t_inputs.T, error * self.__sigmoid_derivative(output))

    def think(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    neural_network = NN()
    print("Random starting weights:\n", neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1],
                                [0, 1, 0],
                                [0, 0, 0]])

    training_outputs = np.array([[0, 1, 1, 0, 0, 0]]).T

    neural_network.train(training_inputs, training_outputs, TRAINING_ITERATIONS)
    print("Post-training weights:\n", neural_network.synaptic_weights)
    print("Considering new situation [1, 0, 0]:\n", neural_network.think(np.array([1, 0, 0])))
