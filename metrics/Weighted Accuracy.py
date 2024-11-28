def weighted_accuracy(answer_key, system_output):
    from collections import Counter
    true_counts = Counter(answer_key.values())  # Total occurrences of each class
    weights = {c: count for c, count in true_counts.items()}  # Use class frequencies as weights

    correct = Counter()
    for qid in answer_key:
        if answer_key[qid] == system_output.get(qid, None):
            correct[answer_key[qid]] += 1

    weighted_sum = sum(weights[c] * correct[c] for c in true_counts)
    total_weighted = sum(weights[c] * true_counts[c] for c in true_counts)

    return weighted_sum / total_weighted
