
def knowledge_model(input_features: dict) -> set:
    """
    A knowledge-based algorithm for predicting patient labels
    :param input_features: A dictionary with all the relevant variables, e.g. BMI, sleep duration
    :return: A set containing integers between 1 and 11
    """
    interim_results = []
    to_return = []
    scores = {}
    scores[1] = score_exercise(input_features['total_exercise'])
    scores[5] = score_sat_fat(sat_fat=input_features['sat_fat'], calories=input_features['calories'])
    scores[6] = score_sugar(input_features['sugar'])
    scores[7] = score_fibre(fibre=input_features['fibre'], gender=input_features['gender'],
                            diabetes=input_features['diabetes'])
    scores[8] = score_BMI(input_features['BMI'])
    scores[9] = score_sleep(input_features['sleep'])
    scores[10] = score_sleep_quality(input_features['sleep_quality'])
    for key, value in scores.items():
        if value > 0:
                interim_results.append({'label': key, 'score': value})
    # Sort biggest to smallest
    interim_results = sorted(interim_results, reverse=True, key=lambda d: d['score'] )
    interim_results = truncate(interim_results)
    # Strip the scores, keep only the labels
    for recs in interim_results:
        to_return.append(recs['label'])
    if len(to_return) == 0:
        to_return.append(11)
    return set(to_return)

def truncate(list_of_recommendations) -> list:
    # Purpose of this function is to remove amber recommendations if the list of recs is bigger than 3
    new_list = []
    if len(list_of_recommendations) > 3:
        for rec in list_of_recommendations:
            if rec['score'] >= 2:
                new_list.append(rec)
        # Maximum 5 recommendations
        if len(new_list) > 5:
            new_list = new_list[0:5]
        return new_list
    else:
        return list_of_recommendations

def score_exercise(total_exercise: float) -> int:
    if total_exercise < 100.0:
        return 2
    elif (total_exercise >= 100) and (total_exercise <= 150):
        return 1
    else:
        return 0

def score_sat_fat(sat_fat: float, calories: float) -> int:
    sat_fat_proportion = sat_fat * 9 / calories
    if sat_fat_proportion > 0.15:
        return 2
    elif (sat_fat_proportion <= 0.15) and (sat_fat_proportion >= 0.1):
        return 1
    else:
        return 0

def score_sugar(sugar: float) -> int:
    if sugar <= 25:
        return 0
    elif (sugar > 25) and (sugar <= 50):
        return 1
    else:
        return 2

def score_fibre(fibre: float, diabetes: int, gender: int) -> int:
    "1 represents diabetes. 1 represents male. Can return -1."
    score = diabetes
    healthy_amount = 31 if gender==1 else 25
    moderate_amount = 25 if gender==1 else 20
    if fibre >= healthy_amount:
        # If the patient eats enough fibre, good
        # Otherwise, diabetes makes this recommendation more important
        score = score - 1
    elif (fibre >= moderate_amount) and (fibre < healthy_amount):
        score += 1
    else:
        score += 2
    return score

def score_BMI(BMI: float) -> int:
    "Returns 1, 2, or 0"
    if BMI >= 30:
        return 2
    elif BMI >= 25.0:
        return 1
    else:
        return 0

def score_sleep(sleep: float) -> int:
    if sleep < 7.5:
        return 2
    elif (sleep >= 7.5) and (sleep < 8.5):
        return 1
    else:
        return 0

def score_sleep_quality(sleep_quality: float) -> int:
    if sleep_quality < 0.5:
        return 2
    elif (sleep_quality >= 0.5) and (sleep_quality < 0.66):
        return 1
    else:
        return 0

print(knowledge_model({'BMI': 19, 'sleep': 7.1, 'sleep_quality': 0.75, 'total_exercise': 160,
                       'sat_fat': 20, 'calories': 2500, 'sugar': 100, 'gender': 1, 'fibre': 32, 'diabetes': 0}))
print(knowledge_model({'BMI': 31, 'sleep': 9.0, 'sleep_quality': 0.4, 'total_exercise': 90,
                       'sat_fat': 25, 'calories': 2200, 'sugar': 24, 'gender': 0, 'fibre': 26, 'diabetes': 1}))