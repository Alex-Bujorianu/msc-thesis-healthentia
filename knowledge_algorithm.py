
def knowledge_model(input_features: dict) -> set:
    """
    A knowledge-based algorithm for predicting patient labels
    :param input_features: A dictionary with all the relevant variables, e.g. BMI, sleep duration
    :return: A set containing integers between 1 and 11
    """
    interim_results = []
    to_return = []
    scores = {}
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
            if rec['score'] == 2:
                new_list.append(rec)
        return new_list
    else:
        return list_of_recommendations

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

print(knowledge_model({'BMI': 19, 'sleep': 7.1, 'sleep_quality': 0.75}))
print(knowledge_model({'BMI': 31, 'sleep': 9.0, 'sleep_quality': 0.4}))