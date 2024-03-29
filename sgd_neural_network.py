from sklearn.neural_network import MLPClassifier
from helper import get_data, partial_accuracy, inverse_transform, \
    partial_accuracy_callable, standardise_data, plot_label_accuracy_cv, \
    count_mismatch_proportion, cross_validate, count_length_ratio
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score, multilabel_confusion_matrix
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

X, Y = get_data("training_set.csv")
# Make sure to scale X – very important!
X = standardise_data(X)
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
                               alpha=0.0005, #best value from the graph
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

print("The mean strict accuracy is ", mean(scores_strict_accuracy), "Stdev is ", stdev(scores_strict_accuracy))
plot_label_accuracy_cv(model_name="SGD Neural Network", model=neural_network, X=X, Y=Y)
print("The proportion of length mismatches is ", -1 * mean(cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(count_mismatch_proportion, greater_is_better=False),
                         cv=cv, n_jobs=-1)))
print("The length ratio is ", mean(cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(count_length_ratio, greater_is_better=True),
                         cv=cv, n_jobs=-1)))
# predictions_cross_val = cross_validate(k=5, X=X, Y=Y, model=MLPClassifier(solver='sgd', activation='relu', learning_rate_init=params['learning_rate_init'],
#                                hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
#                                power_t=params['power_t'],
#                                momentum=params['momentum'],
#                                alpha=0.0005, #best value from the graph
#                                random_state=101))
# partial_accuracy_scores = []
# scores = cross_val_score(neural_network, X, Y,
#                          scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
#                          cv=KFold(n_splits=5, shuffle=False), n_jobs=-1)
# for prediction in predictions_cross_val:
#     partial_accuracy_scores.append(partial_accuracy(inverse_transform(prediction['Truth']),
#                                                     inverse_transform(prediction['Predictions'])))
# print(partial_accuracy_scores)
# print("Scores by sklearn function ", scores)

# Confusion matrix for label 11
neural_network.fit(x_train, y_train)
start = timer()
neural_network.predict(X)
print("Time taken (ms): ", (timer()-start)*1000)
print(multilabel_confusion_matrix(y_true=y_test, y_pred=neural_network.predict(x_test), labels=[10]))