from sklearn.neural_network import MLPClassifier
import numpy as np

class my_nn():
    nn = None
    def __int__(self, depth=1, avg_neurons=100):
        self.nn = MLPClassifier(hidden_layer_sizes=self.generate_neurons(max_depth=depth, avg_neurons=avg_neurons))

    def generate_neurons(self, max_depth, avg_neurons):
        to_return = []
        for i in range(1, max_depth+1):
            to_return.append(int(np.random.normal(avg_neurons, int(0.3*avg_neurons))))
        return tuple(to_return)

    def fit(self, X, Y):
        self.nn.fit(X, Y)