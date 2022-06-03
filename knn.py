from skmultilearn.adapt import MLkNN, BRkNNaClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score
from helper import partial_accuracy, partial_accuracy_callable, \
    count_mismatch_proportion, label_accuracy, get_data, inverse_transform, normalise_data, \
    plot_label_accuracy, plot_label_accuracy_cv, count_length_ratio
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
import json

X, Y = get_data("training_set.csv")
X = normalise_data(X)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)
print(x_test.shape, y_test.shape)

def optimise_mlknn():
    parameters = {'k': range(10, 21), 's': [0.5, 0.7, 1]}
    # Grid search does 5-fold cross-validation
    # Hamming Loss is not in GridSearchCV; you have to pass the function
    classifier = GridSearchCV(MLkNN(), parameters, cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss, greater_is_better=False))
    # No need to split, classifier already does cross-validation
    classifier.fit(X, Y)
    # The negative score is by design, not an error. See: https://stackoverflow.com/questions/44081222/hamming-loss-not-support-in-cross-val-score
    print('best parameters :', classifier.best_params_, 'Best Hamming Loss: ',
          -1 * classifier.best_score_)
    output_file = open("knn_best_params.json", "w")
    json.dump(obj=classifier.best_params_, fp=output_file)

optimise_mlknn()
# Load optimal params.
input_file = open("knn_best_params.json", "r")
params=json.load(input_file)
model = MLkNN(k=params['k'], s=params['s'])
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
scores_partial_accuracy = cross_val_score(model, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print(scores_partial_accuracy)
# We use k-fold validation to have some idea of the variance
# And to ensure there is no overfitting which would be indicated by high variance
print("The mean partial accuracy is is ", mean(scores_partial_accuracy), "\n",
      "The stdev is ", stdev(scores_partial_accuracy))
scores_strict_accuracy = cross_val_score(model, X, Y,
                         scoring=make_scorer(accuracy_score, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean strict accuracy is ", mean(scores_strict_accuracy),
      "The stdev strict accuracy of mlknn is ", stdev(scores_strict_accuracy))
print("The proportion of length mismatches is ", -1 * mean(cross_val_score(model, X, Y,
                         scoring=make_scorer(count_mismatch_proportion, greater_is_better=False),
                         cv=kfold, n_jobs=-1)))
print("The length ratio is ", mean(cross_val_score(model, X, Y,
                         scoring=make_scorer(count_length_ratio, greater_is_better=True),
                         cv=kfold, n_jobs=-1)))
plot_label_accuracy_cv(model_name="MLkNN", model=model, Y=Y, X=X)

# Comparison with binary relevance knn
# Mlknn incorporates MAP
# See docs here: http://scikit.ml/api/skmultilearn.adapt.brknn.html
brknn = BRkNNaClassifier(k=params['k'])
scores_brknn = cross_val_score(brknn, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean partial accuracy of Binary Relevance kNN is ", mean(scores_brknn),
      "\n", "The stdev is ", stdev(scores_brknn))
scores_brknn_strict = cross_val_score(brknn, X, Y,
                         scoring=make_scorer(accuracy_score, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean strict accuracy of Binary Relevance kNN is ", mean(scores_brknn_strict),
      "The stdev is ", stdev(scores_brknn_strict))
print("The proportion of length mismatches (BRknn) is ", -1 * mean(cross_val_score(brknn, X, Y,
                         scoring=make_scorer(count_mismatch_proportion, greater_is_better=False),
                         cv=kfold, n_jobs=-1)))
print("The length ratio (BRknn) is ", mean(cross_val_score(brknn, X, Y,
                         scoring=make_scorer(count_length_ratio, greater_is_better=True),
                         cv=kfold, n_jobs=-1)))
# brknn.fit(x_train, y_train)
# print("A quick sanity check, the length ratio is: ",
#       count_length_ratio(truth=y_test, predictions=brknn.predict(x_test)))
# print("Truth: ", inverse_transform(y_test),
#       "Predictions ", inverse_transform(brknn.predict(x_test)))