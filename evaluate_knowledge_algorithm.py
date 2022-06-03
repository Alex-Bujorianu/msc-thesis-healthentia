from knowledge_algorithm import knowledge_model
from helper import get_data, partial_accuracy, partial_accuracy_callable, \
    count_mismatch_proportion, plot_label_accuracy, \
    strict_accuracy, count_length_ratio
import pandas as pd
from statistics import mean, stdev

X, Y = get_data("training_set.csv", format="pandas")
print(Y)
print(X.head)
X = X.to_dict(orient='list')

predictions = []
for i in range(len(X['calories'])):
    patient = {'BMI': X['bmi'][i],
               'calories': X['calories'][i],
               'sleep': X['Sleep Duration (hr/night)'][i],
               'sleep_quality': X['SleepQuality'][i],
               'total_exercise': X['exercise_duration (min/week)'][i],
               'moderate_exercise': X['fairly_minutes'][i],
               'intense_exercise': X['very_minutes'][i],
               'core_proportion': X['core_involvement'][i],
               'strength_proportion': X['strength_training'][i],
               'sat_fat': X['saturated fats (g/day)'][i],
               'sugar': X['sugar (g/day)'][i],
               'fibre': X['fibre (g/day)'][i],
               'gender': X['gender'][i],
               'diabetes': X['diabetes'][i],
               'chd': X['chd'][i]
               }
    predictor = knowledge_model()
    predictions.append(predictor.predict(patient))

print("Partial accuracy: ", partial_accuracy(predictions, Y))
print("Strict accuracy: ", strict_accuracy(predictions, Y))
print("Mismatch proportions", count_mismatch_proportion(predictions, Y))
print("Length ratio ", count_length_ratio(predictions=predictions, truth=Y))
#print("Predictions: ", predictions, "\n", "Truth: ", Y)
plot_label_accuracy(model_name="Knowledge algorithm", predictions=predictions, truth=Y)