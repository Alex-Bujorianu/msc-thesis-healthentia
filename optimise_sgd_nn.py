from sklearn.neural_network import MLPClassifier
from my_nn import my_nn
from helper import get_data, partial_accuracy, inverse_transform, partial_accuracy_callable, generate_neurons, scale_data
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import json
import os

X, Y = get_data("training_set.csv")
# Make sure to scale X – very important!
X = scale_data(X)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)

cv = KFold(n_splits=5, random_state=101, shuffle=True)

# Let's not generate the NN architecture so randomly
# We want to know if deeper networks perform better or not.
architecture = generate_neurons(max_depth=10, average_neurons_per_layer=100)
parameter_grid = {
    'learning_rate_init': Continuous(0.1, 0.3, distribution='uniform'),
    'hidden_layer_sizes': Categorical(architecture),
    'power_t': Continuous(0.4, 0.8, distribution='uniform'),
    'momentum': Continuous(0.8, 0.95, distribution='uniform')
}

# If you want to use lbfgs, I suggest putting in a separate file.
# You can't use different solvers with the genetic algorithm

# Remember to pass the random state parameter otherwise the performance is all over the place
evolved_nn = MLPClassifier(activation='relu', solver='sgd', learning_rate='invscaling', random_state=101, alpha=0.006)

scorer = make_scorer(partial_accuracy_callable, greater_is_better=True)

evolved_estimator = GASearchCV(estimator=evolved_nn,
                               population_size=10,
                               generations=12,
                              cv=cv,
                              scoring=scorer,
                              param_grid=parameter_grid,
                              n_jobs=os.cpu_count()-1, #leave a core free
                              verbose=True,
                              crossover_probability=0.8,
                              mutation_probability=0.1,
                              criteria='max',
                               )
@ignore_warnings(category=ConvergenceWarning)
def optimise_neural_network():
    evolved_estimator.fit(X, Y)
    print("Best parameters: ", evolved_estimator.best_params_)
    # Save the results
    output_file = open("sgd_nn_best_params.json", "w")
    json.dump(obj=evolved_estimator.best_params_, fp=output_file)
    plot_fitness_evolution(evolved_estimator)
    plt.show()

optimise_neural_network()