from statistics import mean, stdev
import matplotlib.pyplot as plt
from numpy import arange
from sklearn import tree
from sklearn.metrics import hamming_loss, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from helper import partial_accuracy_callable, plot_label_accuracy

from helper import get_data, partial_accuracy, label_accuracy, inverse_transform

X, Y = get_data("training_set.csv")
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)

hamming_loss_train = []
hamming_loss_test = []

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
        hamming_loss_train.append(hamming_loss(y_train, y_train_predictions))
        hamming_loss_test.append(hamming_loss(y_test, y_test_predictions))

    plt.plot(alphas, hamming_loss_train, label="train")
    plt.plot(alphas, hamming_loss_test, label="test")
    plt.legend()
    plt.xlabel("CCP Alpha parameter")
    plt.ylabel("Hamming Loss")
    plt.show()

# Sweet spot seems to be ccp_alpha=0.03
tree_classifier = BinaryRelevance(
        classifier = tree.DecisionTreeClassifier(criterion="gini", ccp_alpha=0.03),
        require_dense = [True, True])
kfold = KFold(n_splits=5, random_state=101, shuffle=True)
tree_classifier.fit(x_train, y_train)
print(type(tree_classifier.predict(x_test)))
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
tree_classifier.fit(x_train, y_train)
predictions_tree = tree_classifier.predict(x_test)

# Per label performance
plot_label_accuracy(model_name="Decision tree", truth=y_test, predictions=predictions_tree)