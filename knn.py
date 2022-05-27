from skmultilearn.adapt import MLkNN, BRkNNaClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score
from helper import partial_accuracy, count_mismatch_proportion, label_accuracy, get_data, inverse_transform, scale_data
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt

X, Y = get_data("training_set.csv")
X = scale_data(X)
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)
print(x_test.shape, y_test.shape)

def optimise_mlknn():
    parameters = {'k': range(10, 21), 's': [0.5, 0.7, 1]}
    # Grid search does 5-fold cross-validation
    # Hamming Loss is not in GridSearchCV; you have to pass the function
    classifier = GridSearchCV(MLkNN(), parameters, cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss, greater_is_better=False))
    print("x train: ", x_train, "\n", "y train: ", y_train)
    # No need to split, classifier already does cross-validation
    classifier.fit(X, Y)
    # The negative score is by design, not an error. See: https://stackoverflow.com/questions/44081222/hamming-loss-not-support-in-cross-val-score
    print('best parameters :', classifier.best_params_, 'Best Hamming Loss: ',
          -1 * classifier.best_score_)

optimise_mlknn()
# Optimal params are k=19, s=0.5
model = MLkNN(k=19, s=0.5)
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
scores = cross_val_score(model, X, Y,
                         scoring=make_scorer(hamming_loss, greater_is_better=True),
                         cv=kfold, n_jobs=-1)

# We use k-fold validation to have some idea of the variance
# And to ensure there is no overfitting which would be indicated by high variance
print("The mean performance is ", mean(scores), "\n",
      "The stdev is ", stdev(scores))

# Now for some intuitive metrics
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("The strict accuracy is ", accuracy_score(y_pred=predictions, y_true=y_test))
print("The partial accuracy of MLkNN is ",
      partial_accuracy(inverse_transform(predictions),
                       inverse_transform(y_test)))
print("The proportion of length mismatches is ",
      count_mismatch_proportion(inverse_transform(predictions), inverse_transform(y_test)))
print(inverse_transform(predictions), "\n", inverse_transform(y_test))

# Per label performance
results = {}
for i in range(1, 12):
    results[str(i)] = label_accuracy(y_true=inverse_transform(y_test),
                                     y_predicted=inverse_transform(predictions), label=i)

names = list(results.keys())
values = list(results.values())
plt.bar(names, values, color='red')
plt.xlabel("Label number")
plt.ylabel("Correctly predicted proportion")
plt.title("MLKNN per-label performance")
plt.show()
# Comparison with binary relevance knn
# Mlknn incorporates MAP
# See docs here: http://scikit.ml/api/skmultilearn.adapt.brknn.html
brknn = BRkNNaClassifier(k=19)
scores = cross_val_score(brknn, X, Y,
                         scoring=make_scorer(hamming_loss, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean Hamming Loss of Binary Relevance kNN is ", mean(scores),
      "\n", "The stdev is ", stdev(scores))
brknn.fit(x_train, y_train)
predictions_br = brknn.predict(x_test)
print("The strict accuracy (Binary Relevance) is ",
      accuracy_score(y_pred=predictions_br, y_true=y_test))
print("The partial accuracy of BR KNN is ",
      partial_accuracy(inverse_transform(predictions_br),
                       inverse_transform(y_test)))