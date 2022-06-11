from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance
from helper import partial_accuracy_callable, plot_label_accuracy_cv, count_mismatch_proportion
from helper import get_data, normalise_data, partial_accuracy, label_accuracy, \
    inverse_transform, cross_validate, count_length_ratio
import json
import math

# SVM is uniquely annoying in that it cannot deal with one-class data
# Label 2 is unused and hence “one class” – we need to prune it
def prune_unused_labels(data: np.ndarray) -> np.ndarray:
    new_data = 0
    for i in range(0, data.shape[1]):
        column = set(data[:, i])
        if 1 not in column:
            new_data = np.delete(data, i, axis=1)
    return new_data

X, Y = get_data("training_set.csv")
Y = prune_unused_labels(Y)
X = normalise_data(X)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)
print(Y.shape)
svm_classifier = BinaryRelevance(
    classifier = SVC(),
    require_dense = [False, True])

svm_classifier.fit(x_train, y_train)
#print("Predictions" , svm_classifier.predict(x_test))

def optimise_svm():
    # don't ask me why we have to use classifier__ before the parameter names.
    # KNN worked fine with the normal names
    parameters = {'classifier__C': np.arange(start=0.5, stop=2, step=0.5), 'classifier__kernel': ['linear', 'poly', 'rbf']}
    # Grid search does 5-fold cross-validation
    classifier = GridSearchCV(svm_classifier, parameters, cv=5, n_jobs=-1, scoring=make_scorer(partial_accuracy_callable, greater_is_better=True))
    # No need to split, classifier already does cross-validation
    classifier.fit(X, Y)
    # The negative score is by design, not an error. See: https://stackoverflow.com/questions/44081222/hamming-loss-not-support-in-cross-val-score
    print('best parameters :', classifier.best_params_, 'Best accuracy: ',
          -1 * classifier.best_score_)
    output_file = open("svm_best_params.json", "w")
    json.dump(obj=classifier.best_params_, fp=output_file)

optimise_svm()
input_file = open("svm_best_params.json", "r")
params=json.load(input_file)
svm_classifier = svm_classifier = BinaryRelevance(
    classifier = SVC(C=params['classifier__C'], kernel=params['classifier__kernel']),
    require_dense = [True, True])
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
scores = cross_val_score(svm_classifier, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
# Because svm fails on some of the splits, we have to replace nans with zeros
# SVM seems to be uniquely bad for our problem
def replace_nans(data:list) -> list:
    new_list = []
    for i in range(len(data)):
        if math.isnan(data[i]):
            new_list.append(0)
        else:
            new_list.append(data[i])
    return new_list

scores = replace_nans(scores)
print("The mean partial accuracy of the SVM is ", mean(scores),
      "\n", "The stdev is ", stdev(scores))