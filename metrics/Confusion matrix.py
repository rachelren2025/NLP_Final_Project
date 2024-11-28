def confusion_matrix(answer_key, system_output):
    from collections import defaultdict

    classes = sorted(set(answer_key.values()))  # All possible classes
    matrix = {true: {pred: 0 for pred in classes} for true in classes}

    for qid in answer_key:
        true = answer_key[qid]
        pred = system_output.get(qid, None)
        if pred is not None:
            matrix[true][pred] += 1

    return matrix


def print_confusion_matrix(matrix):
    classes = sorted(matrix.keys())
    print(" " * 5 + " ".join(f"{c:^5}" for c in classes))
    for true in classes:
        row = [f"{matrix[true][pred]:^5}" for pred in classes]
        print(f"{true:^5} " + " ".join(row))
