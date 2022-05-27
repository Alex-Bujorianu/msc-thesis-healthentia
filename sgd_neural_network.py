from sklearn.neural_network import MLPClassifier
from helper import get_data, partial_accuracy, inverse_transform, partial_accuracy_callable, scale_data
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt


X, Y = get_data("training_set.csv")
# Make sure to scale X – very important!
X = scale_data(X)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)

cv = KFold(n_splits=5, random_state=101, shuffle=True)

input_file = open("sgd_nn_best_params.json", "r")
params=json.load(input_file)
print(params)
neural_network = MLPClassifier(solver='sgd', activation='relu', learning_rate_init=params['learning_rate_init'],
                               hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
                               power_t=params['power_t'],
                               momentum=params['momentum'],
                               alpha=0.006, #seems like a good value from the graph
                               random_state=101)

scores = cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)

def plot_alpha():
    alphas = np.arange(start=0.0001, stop=0.0009, step=0.0001, dtype='float')
    means = []
    stdevs = []
    for alpha in alphas:
        neural_network_loop = MLPClassifier(solver='sgd', activation='relu', learning_rate_init=params['learning_rate_init'],
                                   hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
                                   power_t=params['power_t'],
                                   momentum=params['momentum'],
                                   alpha=alpha,
                                   random_state=101)
        scores_loop = cross_val_score(neural_network_loop, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)
        means.append(mean(scores_loop))
        stdevs.append(stdev(scores_loop))
    plt.plot(alphas, means, label="mean")
    plt.plot(alphas, stdevs, label="stdev")
    plt.legend()
    plt.xlabel("Alpha parameter")
    plt.ylabel("Partial accuracy")
    plt.show()

plot_alpha()

print("The mean partial accuracy is ", mean(scores), "\n",
      "The stdev is ", stdev(scores))

scores_strict_accuracy = cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(accuracy_score, greater_is_better=True),
                         cv=cv, n_jobs=-1)

print("The strict accuracy is ", mean(scores_strict_accuracy))
