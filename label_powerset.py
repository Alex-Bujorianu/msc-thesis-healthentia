from skmultilearn.problem_transform import LabelPowerset
from sklearn import tree
from helper import get_data
from sklearn.model_selection import train_test_split

X, Y = get_data("training_set.csv")
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y, train_size=0.8, random_state=101)

classifier = LabelPowerset(
    classifier = tree.DecisionTreeClassifier(),
    require_dense = [True, True]
)

classifier.fit(x_train, y_train)
print("The number of unique combinations is: ", len(classifier.unique_combinations_))
print("The combinations are: ", classifier.unique_combinations_)
print(classifier.unique_combinations_['10']) #10 is label 11, it counts from 0