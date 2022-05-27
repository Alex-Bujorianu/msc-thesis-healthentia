from sklearn.neural_network import MLPClassifier
from helper import get_data, partial_accuracy, inverse_transform, partial_accuracy_callable, scale_data
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score
from statistics import mean, stdev

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
                               alpha=params['alpha'],
                               random_state=101)

scores = cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(hamming_loss, greater_is_better=False),
                         cv=cv, n_jobs=-1)

print("The mean performance is ", mean(scores), "\n",
      "The stdev is ", stdev(scores))

neural_network.fit(x_train, y_train)
predictions = neural_network.predict(x_test)
print("The strict accuracy is ", accuracy_score(y_pred=predictions, y_true=y_test))
print("The partial accuracy of the SGD neural network is ",
      partial_accuracy_callable(predictions,
                       y_test))