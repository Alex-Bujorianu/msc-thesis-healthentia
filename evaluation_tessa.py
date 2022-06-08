from knowledge_algorithm import knowledge_model
from helper import get_data, partial_accuracy_callable, standardise_data, partial_accuracy
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neural_network import MLPClassifier
import json
from statistics import mean, stdev
from skmultilearn.adapt import MLkNN, BRkNNaClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn import tree
import pandas as pd

X, Y = get_data("training_set_Tessa.csv")
X = standardise_data(X)
input_file = open("sgd_nn_best_params.json", "r")
params=json.load(input_file)
neural_network = MLPClassifier(solver='sgd', activation='relu', learning_rate_init=params['learning_rate_init'],
                               hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
                               power_t=params['power_t'],
                               momentum=params['momentum'],
                               alpha=0.0005, #best value from the graph
                               random_state=101)
cv = KFold(n_splits=5, random_state=101, shuffle=True)
nn_partial_accuracy = cross_val_score(neural_network, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)
print("NN mean partial accuracy ", mean(nn_partial_accuracy), "Standard deviation ", stdev(nn_partial_accuracy))
knn_file = open("knn_best_params.json", "r")
params_knn=json.load(knn_file)
mlknn = MLkNN(k=params_knn['k'], s=params_knn['s'])
mlknn_partial_accuracy = cross_val_score(mlknn, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)
print("MLKNN mean partial accuracy ", mean(mlknn_partial_accuracy), "Standard deviation ", stdev(mlknn_partial_accuracy))
brknn = BRkNNaClassifier(k=params_knn['k'])
scores_brknn = cross_val_score(brknn, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)
print("The mean partial accuracy of Binary Relevance kNN is ", mean(scores_brknn),
      "\n", "The stdev is ", stdev(scores_brknn))

def predict_with_knowledge_model(X: pd.DataFrame) -> list:
    X = X.to_dict(orient='list')
    predictions = []
    for i in range(len(X['calories'])):
        patient = {'BMI': X['bmi'][i],
                   'calories': X['calories'][i],
                   'sleep': X['Sleep Duration (hr/night)'][i],
                   'sleep_quality': X['SleepQuality'][i],
                   'total_exercise': X['exercise_duration (min/week)'][i],
                   'moderate_exercise': X['fairly_minutes'][i],
                   'intense_exercise': X['very_minutes'][i],
                   'core_proportion': X['core_involvement'][i],
                   'strength_proportion': X['strength_training'][i],
                   'sat_fat': X['saturated fats (g/day)'][i],
                   'sugar': X['sugar (g/day)'][i],
                   'fibre': X['fibre (g/day)'][i],
                   'gender': X['gender'][i],
                   'diabetes': X['diabetes'][i],
                   'chd': X['chd'][i]
                   }
        predictor = knowledge_model()
        predictions.append(predictor.predict(patient))
    return predictions

tree_classifier = BinaryRelevance(
        classifier = tree.DecisionTreeClassifier(criterion="gini", ccp_alpha=0.03),
        require_dense = [True, True])
dt_partial_accuracy = cross_val_score(tree_classifier, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=cv, n_jobs=-1)
print("DT mean partial accuracy", mean(dt_partial_accuracy), "Standard deviation ", stdev(dt_partial_accuracy))
X_pd, Y_list = get_data("training_set_Tessa.csv", format="pandas")
predictions = predict_with_knowledge_model(X_pd)
print("Partial accuracy knowledge model: ", partial_accuracy(predictions, Y_list))