import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def partial_accuracy(coder_1, coder_2):
    "This function is commutative: the order doesn’t matter."
    if len(coder_1) != len(coder_2):
        raise Exception("Lengths have to be the same")
    total_accuracy = 0
    for i in range(len(coder_1)):
        subtotal_accuracy = 0
        # Recs can be lists, tuples or sets in the data, but are always converted to sets
        # If we use lists, even if we sort them, the accuracy will be too low because of length inequalities
        coder_1[i] = set(coder_1[i])
        coder_2[i] = set(coder_2[i])
        # print(coder_1[i], coder_2[i])
        for item in coder_1[i]:
            # What if coder 2 gives more recs than coder 1?
            # The code below will still give the correct answer
            # If coder 2 gives 5 recs, coder 1 gives 1 rec, accuracy will be 0.2 if at least one rec matches between the two
            # The result is also correct if coder 1 gives more recs than coder 2
            if item in coder_2[i]:
                subtotal_accuracy += 1 / max(len(coder_1[i]), len(coder_2[i]))
        # print(subtotal_accuracy)
        total_accuracy += subtotal_accuracy
    return total_accuracy / len(coder_1)

def get_data(data_file: str):
    """
    A function that extracts and transforms the data.
    :param data_file: A string representing the filename
    :return: X, a multidimensional numpy array, and Y, a binary numpy array
    """
    training_set = pd.read_csv("Data/" + data_file)
    # Drop nuisance variables
    training_set.drop(labels=['participant_id', 'gender', 'Coder', 'Explanation'], axis=1, inplace=True)
    # Don't forget to transform these variables to 0/1
    training_set['diabetes'] = [1 if x == 'yes' else 0 for x in training_set['diabetes']]
    training_set['chd'] = [1 if x == 'yes' else 0 for x in training_set['chd']]
    X = training_set.drop(labels=['Labels'], axis=1).to_numpy()
    mlb = MultiLabelBinarizer()
    labels = list(range(1, 12))
    mlb.fit([labels])  # why is MLB so retarded? A list of a list? Why?
    Y = [[int(y) for y in x.split(",")] for x in training_set['Labels']]
    Y_transformed = mlb.transform(Y)
    return X, Y_transformed

def count_mismatch_proportion(list1, list2) -> float:
    if len(list1) != len(list2):
        raise TypeError("Length mismatch")
    count = 0
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            count += 1
    return count / len(list1)


def count_labels(label: int, data: list) -> int:
    count = 0
    for iterable in data:
        myset = set(iterable)
        if label in myset:
            count += 1
    return count


def label_accuracy(y_true: list, y_predicted: list, label: int) -> float:
    """
    :param y_true: The ground truth, formatted as a list of iterable objects
    :param y_predicted: The predictions, formatted as a list of iterable objects (ideally sets)
    :param label: An integer representing the label
    :return: The proportion of correct matches for the given label
    """
    if len(y_true) != len(y_predicted):
        raise TypeError("Must be the same length")
    count_true = 0
    count_predicted = 0
    for i in range(len(y_true)):
        set_true = set(y_true[i])
        set_predicted = set(y_predicted[i])
        if label in set_true:
            count_true += 1
            if label in set_predicted:
                count_predicted += 1
    if count_true > 0:
        return count_predicted / count_true
    else:
        return 0
