def rmse(answer_key, system_output):
    import math
    total_squared_error = sum((answer_key[qid] - system_output.get(qid, 0))**2 for qid in answer_key)
    total = len(answer_key)
    return math.sqrt(total_squared_error / total)
