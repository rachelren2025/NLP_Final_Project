from collections import Counter

def calculate_f1(true_labels, predicted_labels):
    # Calculate True Positives, False Positives, and False Negatives for each class
    classes = set(true_labels + predicted_labels)
    tp = Counter()  # True Positives
    fp = Counter()  # False Positives
    fn = Counter()  # False Negatives

    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            tp[true] += 1  # Correct prediction
        else:
            fp[pred] += 1  # Predicted as this class but actually incorrect
            fn[true] += 1  # Missed this class in the prediction

    # Calculate Precision, Recall, and F1 for each class
    precisions, recalls, f1_scores = {}, {}, {}
    for c in classes:
        precisions[c] = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] > 0 else 0
        recalls[c] = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] > 0 else 0
        if precisions[c] + recalls[c] > 0:
            f1_scores[c] = 2 * (precisions[c] * recalls[c]) / (precisions[c] + recalls[c])
        else:
            f1_scores[c] = 0

    # Macro F1
    macro_f1 = sum(f1_scores.values()) / len(classes)

    # Micro F1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1 = 0

    return macro_f1, micro_f1
