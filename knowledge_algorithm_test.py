import unittest
from knowledge_algorithm import knowledge_model

# These are just prima facie tests
# to ensure there are no programming errors when modifying the knowledge algorithm
# Use the data for an actual evaluation

class TestSum(unittest.TestCase):

    def test_patient_1(self):
        patient_1 = {'BMI': 19, 'sleep': 7.1, 'sleep_quality': 0.75, 'total_exercise': 160,
                       'moderate_exercise': 50, 'intense_exercise': 20,
                       'sat_fat': 20, 'calories': 2500, 'sugar': 100, 'gender': 1, 'fibre': 32, 'diabetes': 0,
                       'core_proportion': 0.35, 'strength_proportion': 0.42, 'chd': 0}
        predictor = knowledge_model()
        self.assertEqual(predictor.predict(input_features=patient_1), {9, 2, 6})

    def test_patient_2(self):
        patient_2 = {'BMI': 31, 'sleep': 9.0, 'sleep_quality': 0.4, 'total_exercise': 90,
                       'moderate_exercise': 30, 'intense_exercise': 10,
                       'sat_fat': 25, 'calories': 2200, 'sugar': 24, 'gender': 0, 'fibre': 26, 'diabetes': 1,
                       'core_proportion': 0.2, 'strength_proportion': 0.3, 'chd': 0}
        predictor = knowledge_model()
        self.assertEqual(predictor.predict(input_features=patient_2), {8, 1, 10})

    def test_patient_3(self):
        patient_3 = {'BMI': 23, 'sleep': 9.0, 'sleep_quality': 0.8, 'total_exercise': 90,
                     'moderate_exercise': 30, 'intense_exercise': 10,
                     'sat_fat': 30, 'calories': 2000, 'sugar': 24, 'gender': 0, 'fibre': 15, 'diabetes': 0,
                     'core_proportion': 0.2, 'strength_proportion': 0.3, 'chd': 1}
        predictor = knowledge_model()
        self.assertEqual(predictor.predict(input_features=patient_3), {5, 1, 7})

if __name__ == '__main__':
    unittest.main()
