import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance
from statistics import mean
from helper import partial_accuracy

training_set_100 = pd.read_csv("Data/training_set_100.csv")
training_set_Harm = pd.read_csv("Data/training_set_Harm.csv")
training_set_Miriam = pd.read_csv("Data/training_set_Miriam.csv")

labels_alex = training_set_100['Labels'].tolist()[0:50]
labels_Harm = training_set_Harm['Labels'].tolist()[0:50]
labels_Miriam = training_set_Miriam['Labels'].tolist()[0:50]
# We have to compare sets of labels, not strings
labels_alex = [set([int(y) for y in x.split(",")]) for x in labels_alex]
labels_Harm = [set([int(y) for y in x.split(",")]) for x in labels_Harm]
labels_Miriam = [set([int(y) for y in x.split(",")]) for x in labels_Miriam]
#print(labels_alex, "\n", labels_Harm, "\n", labels_Miriam)
# Data should have the following format:
# List of 3-tuples
# Each tuple represents coder, item (patient), label (as frozen set)
task_data = []
for i in range(len(labels_alex)):
    alex_tuple = tuple(['Alex', 'Patient' + str(i), frozenset(labels_alex[i])])
    harm_tuple = tuple(['Harm', 'Patient' + str(i), frozenset(labels_Harm[i])])
    miriam_tuple = tuple(['Miriam', 'Patient' + str(i), frozenset(labels_Miriam[i])])
    task_data.append(alex_tuple)
    task_data.append(harm_tuple)
    task_data.append(miriam_tuple)

task = AnnotationTask(distance = masi_distance)
task.load_array(task_data)
print("Krippendorff's alpha", task.alpha())

# Transformations – this MLB is a pain in the ass
mlb = MultiLabelBinarizer()
# Problem: Harm has used more recommendations than I have, therefore fit_transform doesn't work
# Harm has used 10 labels, I have used 9, so we end up with inconsistent shapes
# Solution is to first fit on all the possible labels, which are known
labels = list(range(1, 12))
mlb.fit([labels]) #why is MLB so retarded? A list of a list? Why?
labels_alex_transformed = mlb.transform(labels_alex)
labels_Harm_transformed = mlb.transform(labels_Harm)
labels_Miriam_transformed = mlb.transform(labels_Miriam)
#print([metrics.hamming_loss(labels_alex_transformed, labels_Harm_transformed),
                                       #metrics.hamming_loss(labels_alex_transformed, labels_Miriam_transformed)])
print("The mean Hamming Loss is", mean([metrics.hamming_loss(labels_alex_transformed, labels_Harm_transformed),
                                       metrics.hamming_loss(labels_alex_transformed, labels_Miriam_transformed),
                                        metrics.hamming_loss(labels_Harm_transformed, labels_Miriam_transformed)]))
print("The mean strictly matching accuracy is ", mean([metrics.accuracy_score(labels_alex_transformed, labels_Harm_transformed),
                                                      metrics.accuracy_score(labels_alex_transformed, labels_Miriam_transformed),
                                                       metrics.accuracy_score(labels_Harm_transformed, labels_Miriam_transformed)]))



print("The mean partial accuracy is ", mean([partial_accuracy(labels_alex, labels_Harm),
                                            partial_accuracy(labels_alex, labels_Miriam),
                                             partial_accuracy(labels_Miriam, labels_Harm)]))
#print("Is the partial accuracy function commutative? See for yourself: ", partial_accuracy(labels_Harm, labels_alex))