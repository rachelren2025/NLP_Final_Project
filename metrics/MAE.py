def mean_absolute_error(answer_key, system_output):
    total_error = sum(abs(answer_key[qid] - system_output.get(qid, 0)) for qid in answer_key)
    total = len(answer_key)
    return total_error / total

