import pandas as pd
import krippendorff
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

training_set_100 = pd.read_csv("training_set_100.csv")
training_set_Harm = pd.read_csv("training_set_Harm.csv")

labels_alex = training_set_100['Labels'].tolist()[0:50]
labels_Harm = training_set_Harm['Labels'].tolist()[0:50]
print(len(labels_alex))
# #List of lists needed
# reliability_data = [labels_alex, labels_Harm]
# #Turn strings into list of ints
# new_reliability_data = []
# for coder in reliability_data:
#     new_coder = []
#     for string in coder:
#         splitted_list = string.split(",")
#         splitted_list_ints = [int(v) for v in splitted_list]
#         new_coder.append(splitted_list_ints)
#     new_reliability_data.append(new_coder)
#
# print(new_reliability_data)
# # Krippendorff’s alpha can also be low if the expected agreement is high (observed - expected)
# # or if the number of datapoints is small (because agreement can happen by chance)
# print("And the Krippendorff’s alpha is…",
#       krippendorff.alpha(reliability_data=new_reliability_data, value_domain=list(range(1, 12)), level_of_measurement="nominal"))

# Transformations – this MLB is a pain in the ass
mlb = MultiLabelBinarizer()
labels_alex = [set([int(y) for y in x.split(",")]) for x in labels_alex]
labels_Harm = [set([int(y) for y in x.split(",")]) for x in labels_Harm]
# Problem: Harm has used more recommendations than I have, therefore fit_transform doesn't work
# Harm has used 10 labels, I have used 9, so we end up with inconsistent shapes
# Solution is to first fit on all the possible labels, which are known
mlb.fit([list(range(1, 12))]) #why is MLB so retarded? A list of a list? Why?
labels_alex_transformed = mlb.transform(labels_alex)
labels_Harm_transformed = mlb.transform(labels_Harm)
print(labels_alex_transformed.shape)
print(labels_Harm_transformed.shape)
print("The Hamming Loss is", metrics.hamming_loss(labels_alex_transformed, labels_Harm_transformed))
print("The strictly matching accuracy is ", metrics.accuracy_score(labels_alex_transformed, labels_Harm_transformed))


def partial_accuracy(coder_1, coder_2):
    "Coder_1 is the ground truth coder."
    if len(coder_1) != len(coder_2):
        raise Exception("Lengths have to be the same")
    total_accuracy = 0
    for i in range(len(coder_1)):
        subtotal_accuracy = 0
        # Recs can be lists, tuples or sets in the data, but are always converted to sets
        # If we use lists, even if we sort them, the accuracy will be too low because of length inequalities
        coder_1[i] = set(coder_1[i])
        coder_2[i] = set(coder_2[i])
        print(coder_1[i], coder_2[i])
        for item in coder_1[i]:
            # What if coder 2 gives more recs than coder 1?
            # The code below will still give the correct answer
            # If coder 2 gives 5 recs, coder 1 gives 1 rec, accuracy will be 0.2 if at least one rec matches between the two
            try:
                if item in coder_2[i]:
                    # If coder 2 does not give enough recs, the accuracy will not be 1
                    subtotal_accuracy += 1/len(coder_1[i])
            except IndexError:
                # Coder 1 has given more recommendations than coder 2
                break
        print(subtotal_accuracy)
        total_accuracy += subtotal_accuracy
    return total_accuracy / len(coder_1)

print("The partial accuracy is ", partial_accuracy(labels_alex, labels_Harm))