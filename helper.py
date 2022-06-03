import numpy as np
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from statistics import mean

mlb = MultiLabelBinarizer()
labels = list(range(1, 12))
mlb.fit([labels])

def generate_neurons(max_depth: int, average_neurons_per_layer: int) -> list:
    """
    This function is used to generate a Categorical list for use with the genetic algorithm.
    :param max_depth: The maximum number of hidden layers. The function will progressively create deeper networks.
    :param average_neurons_per_layer: self-explanatory. A normal distribution is used.
    :return: A list of tuples, of variable length, where each number in the tuple refers to the number of neurons at the ith hidden layer
    """
    to_return = []
    for i in range(1, max_depth+1):
        my_tuple = tuple()
        for j in range(1, i+1):
            my_tuple = my_tuple + ((max(1, int(np.random.normal(average_neurons_per_layer, int(0.3*average_neurons_per_layer))))),)
        to_return.append(my_tuple)
    return to_return

def normalise_data(X: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(copy=True)
    scaler.fit(X)
    return scaler.transform(X)

def standardise_data(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler(copy=True)
    # Figure our how to exclude gender, diabetes and chd from the scaling
    print(X[0][7], X[0][8], X[0][9])
    scaler.fit(X)
    return scaler.transform(X)

def strict_accuracy(coder_1, coder_2):
    if len(coder_1) != len(coder_2):
        raise Exception("Lengths have to be the same")
    count_correct = 0
    for i in range(len(coder_1)):
        coder_1[i] = set(coder_1[i])
        coder_2[i] = set(coder_2[i])
        if coder_1[i] == coder_2[i]:
            count_correct += 1
    return count_correct / len(coder_1)

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

def partial_accuracy_callable(coder_1, coder_2):
    """For use with make_scorer.
     @param coder_1: either a numpy array generated by the MultiLabelBinarizer,
     or a scipy sparse matrix.
     @param coder_2: same type as above """
    # Nothing can be simple in life,
    # with numpy arrays and two THREE different kinds of scipy sparse matrices being used interchangeably.
    # What a mess.
    if (type(coder_1) == scipy.sparse._lil.lil_matrix) \
            or (type(coder_1) == scipy.sparse._csr.csr_matrix)\
            or (type(coder_1) == scipy.sparse._csc.csc_matrix):
        coder_1 = coder_1.toarray()
    if (type(coder_2) == scipy.sparse._lil.lil_matrix) \
            or (type(coder_2) == scipy.sparse._csr.csr_matrix)\
            or (type(coder_2) == scipy.sparse._csc.csc_matrix):
        coder_2 = coder_2.toarray()
    # Make sure this code runs after the typecasting otherwise it will fail
    # Scipy matrices can fuck off
    if len(coder_1) != len(coder_2):
        raise Exception("Lengths have to be the same")
    coder_1 = inverse_transform(coder_1)
    coder_2 = inverse_transform(coder_2)
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

def get_data(data_file: str, format="numpy"):
    """
    A function that extracts and transforms the data.
    :param data_file: A string representing the filename
    :return: X, a multidimensional numpy array (if numpy is chosen; else pandas dataframe),
    and Y, a binary numpy array (else list of lists)
    """
    training_set = pd.read_csv("Data/" + data_file)
    # Drop nuisance variables
    training_set.drop(labels=['participant_id', 'Coder', 'Explanation'], axis=1, inplace=True)
    # Don't forget to transform these variables to 0/1
    training_set['diabetes'] = [1 if x == 'yes' else 0 for x in training_set['diabetes']]
    training_set['chd'] = [1 if x == 'yes' else 0 for x in training_set['chd']]
    training_set['gender'] = [1 if x == 'male' else 0 for x in training_set['gender']]
    X = training_set.drop(labels=['Labels'], axis=1)
    Y = [[int(y) for y in x.split(",")] for x in training_set['Labels']]
    if format=="numpy":
        Y = mlb.transform(Y)
        X = X.to_numpy()
    return X, Y

def count_mismatch_proportion(list1, list2) -> float:
    if type(list1) != list:
        list1 = inverse_transform(list1)
    if type(list2) != list:
        list2 = inverse_transform(list2)
    if len(list1) != len(list2):
        raise TypeError("Length mismatch")
    count = 0
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            count += 1
    return count / len(list1)

def count_length_ratio(truth, predictions) -> float:
    """
    Counts the ratio of the length of recommendations in one list compared to the other.
    Useful to see whether an algorithm is giving too many or too few recommendations (or both)
    A ratio above 1 indicates too many recommendations, < 1 too few, and 1 both.
    """
    # The following typecasts are required to work with sklearn's cross-validation
    if type(truth) != list:
        truth = inverse_transform(truth)
    if type(predictions) != list:
        predictions = inverse_transform(predictions)
    if len(truth) != len(predictions):
        raise TypeError("Length mismatch")
    total_truth = 0
    total_predictions = 0
    for i in range(len(truth)):
        total_truth += len(truth[i])
        total_predictions += len(predictions[i])
    return total_predictions / total_truth

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

def inverse_transform(to_transform: list) -> list:
    "Inverse transform from MLB"
    return mlb.inverse_transform(to_transform)

def plot_label_accuracy(model_name: str, truth: np.ndarray, predictions: np.ndarray):
    # Per label performance
    results = {}
    for i in range(1, 12):
        results[str(i)] = label_accuracy(y_true=inverse_transform(truth) if type(truth)!=list else truth,
                                         y_predicted=inverse_transform(predictions) if type(predictions)!=list else predictions, label=i)

    names = list(results.keys())
    values = list(results.values())
    plt.bar(names, values, color='red')
    plt.xlabel("Label number")
    plt.ylabel("Correctly predicted proportion")
    plt.title(model_name + " per-label performance")
    plt.show()

def plot_label_accuracy_cv(model, X, Y, model_name: str):
    """
    Like the function above, but includes cross-validation
    :param model: An sklearn-like model
    :param X: The input features
    :param Y: The labels
    :return: Plots a graph
    """
    list_of_cv_predictions = cross_validate(k=5, X=X, Y=Y, model=model)
    list_of_results = []
    for dictionary in list_of_cv_predictions:
        results = {}
        for i in range(1, 12):
            results[str(i)] = label_accuracy(y_true=inverse_transform(dictionary['Truth']),
                                         y_predicted=inverse_transform(dictionary['Predictions']), label=i)
        list_of_results.append(results)

    print(list_of_results)
    final_results = {}
    for i in range(1, 12):
        final_results[str(i)] = mean([d[str(i)] for d in list_of_results])
    print(final_results)
    values = list(final_results.values())
    names = list(final_results.keys())
    plt.bar(names, values, color='red')
    plt.xlabel("Label number")
    plt.ylabel("Correctly predicted proportion")
    plt.title(model_name + " per-label performance")
    plt.show()

def cross_validate(k: int, X: np.ndarray, Y: np.ndarray, model) -> list:
    """
    This function is almost but not exactly identical to KFold from sklearn.
    It is intended for use by plot_label_accuracy_cv()
    :param k: The k in k-fold.
    :param X: The numpy array representing the input features
    :param Y: The numpy array representing the correct output, as a binary mask
    :param model: An sklearn-like model
    :return: A list of predictions (not scores!) of length k
    """
    if type(k) != int:
        raise TypeError("k needs to be an integer")
    if X.shape[0] != Y.shape[0]:
        raise TypeError("Lengths need to be the same")
    print(X.shape[0], Y.shape[0])
    if X.shape[0] % k != 0:
        raise Exception("Pick a more sensible k value. The chosen k value does not divide the data evenly.")
    length = X.shape[0]
    split_number = int(length / k)
    start = 0
    predictions = []
    # The annoying range function is exclusive of the last number
    for i in range(split_number, X.shape[0]+split_number, split_number):
        X_test = X[start:i]
        X_train = np.concatenate((X[i:length], X[0:start]))
        # print("Length of first part: ", X[i:length].shape[0], "Length of second part ", X[0:start].shape[0])
        # print("Length of X train: ", X_train.shape[0])
        # print("Length of X test: ", X_test.shape[0])
        Y_test = Y[start:i]
        Y_train = np.concatenate((Y[i:length], Y[0:start]))
        start += split_number
        model.fit(X_train, Y_train)
        predictions.append({"Predictions": model.predict(X_test), "Truth": Y_test})
    assert len(predictions) == k
    return predictions
