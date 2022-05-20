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
        #print(coder_1[i], coder_2[i])
        for item in coder_1[i]:
            # What if coder 2 gives more recs than coder 1?
            # The code below will still give the correct answer
            # If coder 2 gives 5 recs, coder 1 gives 1 rec, accuracy will be 0.2 if at least one rec matches between the two
            # The result is also correct if coder 1 gives more recs than coder 2
            if item in coder_2[i]:
                subtotal_accuracy += 1/max(len(coder_1[i]), len(coder_2[i]))
        #print(subtotal_accuracy)
        total_accuracy += subtotal_accuracy
    return total_accuracy / len(coder_1)

def count_mismatch_proportion(list1, list2) -> float:
    if len(list1) != len(list2):
        raise TypeError("Length mismatch")
    count = 0
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            count += 1
    return count / len(list1)

def count_labels(label: int, data: list) -> int:
    count = 0
    for iterable in data:
        myset = set(iterable)
        if label in myset:
            count += 1
    return count