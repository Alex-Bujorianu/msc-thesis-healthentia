from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import hamming_loss, make_scorer
import pandas as pd
import numpy as np

#Extract and transform data into the right format
training_set = pd.read_csv("training_set.csv")
# Drop nuisance variables
training_set.drop(labels=['participant_id', 'gender', 'Coder', 'Explanation'], axis=1, inplace=True)
# Don't forget to transform these variables to 0/1
training_set['diabetes'] = [1 if x=='yes' else 0 for x in training_set['diabetes']]
training_set['chd'] = [1 if x=='yes' else 0 for x in training_set['chd']]
print(training_set)
X = training_set.drop(labels=['Labels'], axis=1).to_numpy()
print("X is: ", X)
mlb = MultiLabelBinarizer()
labels = list(range(1, 12))
mlb.fit([labels]) #why is MLB so retarded? A list of a list? Why?
Y = [[int(y) for y in x.split(",")] for x in training_set['Labels']]
Y_transformed = mlb.transform(Y)
print(Y_transformed)
x_train, x_test = train_test_split(X, test_size=0.8, random_state=101)
y_train, y_test = train_test_split(Y_transformed, test_size=0.8, random_state=101)

def optimise_mlknn():
    parameters = {'k': range(10, 21), 's': [0.5, 0.7, 1]}
    # Grid search does 5-fold cross-validation
    # Hamming Loss is not in GridSearchCV; you have to pass the function
    classifier = GridSearchCV(MLkNN(), parameters, cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss, greater_is_better=False))
    print("x train: ", x_train, "\n", "y train: ", y_train)
    # No need to split, classifier already does cross-validation
    classifier.fit(X, Y_transformed)
    # The negative score is by design, not an error. See: https://stackoverflow.com/questions/44081222/hamming-loss-not-support-in-cross-val-score
    print('best parameters :', classifier.best_params_, 'Best Hamming Loss: ',
          -1 * classifier.best_score_)

optimise_mlknn()