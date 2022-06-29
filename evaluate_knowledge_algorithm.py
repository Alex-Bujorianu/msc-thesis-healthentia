from knowledge_algorithm import knowledge_model
from helper import get_data, partial_accuracy, partial_accuracy_callable, \
    count_mismatch_proportion, plot_label_accuracy, \
    strict_accuracy, count_length_ratio, transform
import pandas as pd
from statistics import mean, stdev
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from timeit import default_timer as timer

X, Y = get_data("training_set.csv", format="pandas")
print(Y)
print(X.head)

def predict_with_knowledge_model(X: pd.DataFrame) -> list:
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
    return predictions

start = timer()
predictions = predict_with_knowledge_model(X)
print("Time taken (ms): ", (timer()-start)*1000)
print("Partial accuracy: ", partial_accuracy(predictions, Y))
print("Strict accuracy: ", strict_accuracy(predictions, Y))
print("Mismatch proportions", count_mismatch_proportion(predictions, Y))
print("Length ratio ", count_length_ratio(predictions=predictions, truth=Y))
#print("Predictions: ", predictions, "\n", "Truth: ", Y)
plot_label_accuracy(model_name="Knowledge algorithm", predictions=predictions, truth=Y)

X_np, Y_np = get_data("training_set.csv")
y_train, y_test = train_test_split(Y_np, train_size=0.8, random_state=101)
# x will be a panda df
x_train, x_test = train_test_split(X, train_size=0.8, random_state=101)
test_predictions = predict_with_knowledge_model(x_test)
print(multilabel_confusion_matrix(y_true=y_test, y_pred=transform(test_predictions), labels=[10]))