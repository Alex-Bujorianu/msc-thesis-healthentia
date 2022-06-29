from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance
from helper import partial_accuracy_callable, plot_label_accuracy_cv, count_mismatch_proportion
from helper import get_data, normalise_data, partial_accuracy, label_accuracy, \
    inverse_transform, cross_validate, count_length_ratio, prune_unused_labels
import json
import math
from timeit import default_timer as timer

X, Y = get_data("training_set.csv")
print("Y before pruning ", Y.shape)
# Label 4 has a single true value, we need to remove it manually
# Make sure to do this BEFORE calling prune_unused_labels because the shape changes
Y_pruned = np.delete(Y, 3, axis=1)
Y_pruned = prune_unused_labels(Y_pruned)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y_pruned, train_size=0.8, random_state=101)
print(Y_pruned.shape)
X = normalise_data(X)
svm_classifier = BinaryRelevance(
    classifier = SVC(),
    require_dense = [False, True])

#print("Predictions" , svm_classifier.predict(x_test))

def optimise_svm():
    # don't ask me why we have to use classifier__ before the parameter names.
    # KNN worked fine with the normal names
    parameters = {'classifier__C': np.arange(start=0.5, stop=2, step=0.5), 'classifier__kernel': ['linear', 'poly', 'rbf']}
    # Grid search does 5-fold cross-validation
    classifier = GridSearchCV(svm_classifier, parameters, cv=5, n_jobs=-1, scoring=make_scorer(partial_accuracy_callable, greater_is_better=True))
    # No need to split, classifier already does cross-validation
    classifier.fit(X, Y_pruned)
    # The negative score is by design, not an error. See: https://stackoverflow.com/questions/44081222/hamming-loss-not-support-in-cross-val-score
    print('best parameters :', classifier.best_params_, 'Best accuracy: ',
          -1 * classifier.best_score_)
    output_file = open("svm_best_params.json", "w")
    json.dump(obj=classifier.best_params_, fp=output_file)

input_file = open("svm_best_params.json", "r")
params=json.load(input_file)
svm_classifier = svm_classifier = BinaryRelevance(
    classifier = SVC(C=params['classifier__C'], kernel=params['classifier__kernel']),
    require_dense = [True, True])
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
scores = cross_val_score(svm_classifier, X, Y_pruned,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=kfold, n_jobs=-1)

# This function is no longer needed
def replace_nans(data:list) -> list:
    new_list = []
    for i in range(len(data)):
        if math.isnan(data[i]):
            new_list.append(0)
        else:
            new_list.append(data[i])
    return new_list

print("The mean partial accuracy of the SVM is ", mean(scores),
      "\n", "The stdev is ", stdev(scores))
scores_strict = cross_val_score(svm_classifier, X, Y_pruned,
                         scoring=make_scorer(accuracy_score, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean strict accuracy is ", mean(scores_strict), "Stdev ", stdev(scores_strict))
print("The proportion of length mismatches is ", -1 * mean(cross_val_score(svm_classifier, X, Y_pruned,
                         scoring=make_scorer(count_mismatch_proportion, greater_is_better=False),
                         cv=kfold, n_jobs=-1)))
print("The mean length ratio is ", mean(cross_val_score(svm_classifier, X, Y_pruned,
                         scoring=make_scorer(count_length_ratio, greater_is_better=True),
                         cv=kfold, n_jobs=-1)))
# We need to use special workarounds because labels 2 and 4 are missing
plot_label_accuracy_cv(svm_classifier, X=X, Y=Y_pruned, model_name="SVM", offset=[1, 2])
svm_classifier.fit(x_train, y_train)
start = timer()
svm_classifier.predict(X)
print("Time taken (ms): ", (timer()-start)*1000)
print("SVM confusion matrix", multilabel_confusion_matrix(y_true=y_test, y_pred=svm_classifier.predict(x_test), labels=[8]))
