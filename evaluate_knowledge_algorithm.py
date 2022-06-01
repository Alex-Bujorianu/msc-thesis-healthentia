from knowledge_algorithm import knowledge_model
from helper import get_data, partial_accuracy
import pandas as pd

X, Y = get_data("training_set.csv", format="pandas")

print(X.head)
X = X.to_dict(orient='list')


for i in range(len(X['calories'])):
    patient = {'calories': X['calories'][i],
               'sleep': X['Sleep Duration (hr/night)'][i],
               'sleep_quality': X['SleepQuality'][i]}