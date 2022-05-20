import matplotlib.pyplot as plt
from helper import count_labels
import pandas as pd

column = pd.read_csv("training_set.csv")['Labels']
labels = [set([int(y) for y in x.split(",")]) for x in column]

data = {}
for i in range(1, 12):
    data[str(i)] = count_labels(data=labels, label=i)

print(data)
names = list(data.keys())
values = list(data.values())

plt.bar(names, values, color='green')
plt.xlabel("Label number")
plt.ylabel("Label count")
plt.title("Unbalanced data")
plt.show()