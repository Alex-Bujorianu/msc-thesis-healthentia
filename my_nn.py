from sklearn.neural_network import MLPClassifier
import numpy as np

# This class was intended to allow optimising the depth and number of neurons separately using the genetic algorithm.
# Unfortunately, it is not compatible with the clone() method that sklearn_genetic uses.
class my_nn(MLPClassifier):
    def __init__(self, depth=10, avg_neurons=100, activation='relu', solver='sgd', learning_rate='invscaling',
                 learning_rate_init=0.1, power_t=0.5, alpha=0.0001, momentum=0.9):
       super().__init__(hidden_layer_sizes=self.generate_neurons(depth=depth, avg_neurons=avg_neurons),
                        activation=activation, solver=solver, learning_rate=learning_rate,
                        learning_rate_init=learning_rate_init, power_t=power_t, alpha=alpha, momentum=momentum)

    def generate_neurons(self, depth, avg_neurons):
        to_return = []
        for i in range(1, depth+1):
            to_return.append(int(np.random.normal(avg_neurons, int(0.3*avg_neurons))))
        return tuple(to_return)

    def get_params(self, deep=True):
        return super().get_params(deep=deep)