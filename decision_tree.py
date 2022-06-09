from statistics import mean, stdev
import matplotlib.pyplot as plt
from numpy import arange
from sklearn import tree
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from helper import partial_accuracy_callable, plot_label_accuracy_cv, count_mismatch_proportion
from helper import get_data, partial_accuracy, label_accuracy, \
    inverse_transform, cross_validate, count_length_ratio

X, Y = get_data("training_set.csv")
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)

partial_accuracy_train = []
partial_accuracy_test = []

tree_classifier = BinaryRelevance(
    classifier = tree.DecisionTreeClassifier(criterion="Gini"),
    require_dense = [True, True])

def prune_tree():
    # Find the right value of ccp_alpha to prevent overfitting by pruning
    alphas = arange(start=0.01, stop=0.15, step=0.01)
    for alpha in alphas:
        tree_classifier = BinaryRelevance(
        classifier = tree.DecisionTreeClassifier(criterion="gini", ccp_alpha=alpha),
        require_dense = [True, True])
        tree_classifier.fit(x_train, y_train)
        y_train_predictions = tree_classifier.predict(x_train)
        y_test_predictions = tree_classifier.predict(x_test)
        partial_accuracy_train.append(partial_accuracy_callable(y_train, y_train_predictions))
        partial_accuracy_test.append(partial_accuracy_callable(y_test, y_test_predictions))

    plt.plot(alphas, partial_accuracy_train, label="train")
    plt.plot(alphas, partial_accuracy_test, label="test")
    plt.legend()
    plt.xlabel("CCP Alpha parameter")
    plt.ylabel("Partial accuracy")
    plt.show()
prune_tree()
# Sweet spot seems to be ccp_alpha=0.03
tree_classifier = BinaryRelevance(
        classifier = tree.DecisionTreeClassifier(criterion="gini", ccp_alpha=0.01),
        require_dense = [True, True])
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
scores = cross_val_score(tree_classifier, X, Y,
                         scoring=make_scorer(partial_accuracy_callable, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean partial accuracy of the BR decision tree is ", mean(scores),
      "\n", "The stdev is ", stdev(scores))
scores_strict = cross_val_score(tree_classifier, X, Y,
                         scoring=make_scorer(accuracy_score, greater_is_better=True),
                         cv=kfold, n_jobs=-1)
print("The mean strict accuracy of the decision tree is ", mean(scores_strict),
      "The stdev of the strict accuracy is ", stdev(scores_strict))
# Per label performance, now with cross-validation!
plot_label_accuracy_cv(model_name="Decision tree", model=tree_classifier, Y=Y, X=X)

print("The proportion of length mismatches is ", -1 * mean(cross_val_score(tree_classifier, X, Y,
                         scoring=make_scorer(count_mismatch_proportion, greater_is_better=False),
                         cv=kfold, n_jobs=-1)))
print("The length ratio is ", mean(cross_val_score(tree_classifier, X, Y,
                         scoring=make_scorer(count_length_ratio, greater_is_better=True),
                         cv=kfold, n_jobs=-1)))
tree_classifier.fit(x_train, y_train)
print(tree_classifier.classifiers_)
print("The last tree has ", tree_classifier.classifiers_[10].get_n_leaves(), " nodes")
print("The last tree has ", tree_classifier.classifiers_[10].get_depth(), " depth")
# Tree for label 11
tree.plot_tree(tree_classifier.classifiers_[10])
# Tree for first label
tree.plot_tree(tree_classifier.classifiers_[0])
plt.show()
print("DT confusion matrix", multilabel_confusion_matrix(y_true=y_test, y_pred=tree_classifier.predict(x_test), labels=[10]))