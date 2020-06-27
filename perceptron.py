import numpy as np
SEED = 4

np.set_printoptions(formatter={'all': lambda x: str(x)})

class NN:

    def __init__(self, seed=SEED):
        np.random.seed(seed)
        # Give the synapses random weightings initially
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1 - self. __sigmoid(x))

    def train(self, t_inputs, t_outputs, iters):
        for iterations in range(iters):
            # Calculate what the output could be given current understanding (weightings)
            output = self.think(t_inputs)
            # See how much the guess differs from the actual answer
            error = t_outputs - output
            # Improve understanding of the data by changing the weighting values of synapses
            self.synaptic_weights += np.dot(t_inputs.T, error * self.__sigmoid_derivative(output))

    def think(self, inputs):
        # Calculate a guessed output as a product of the inputs and weights
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    TRAINING_ITERATIONS = 100000

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
