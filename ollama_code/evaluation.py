import pickle
from collections import Counter

model = "llama3.1"

def load_dict_files(output_file, key_file):
    with open(output_file, "rb") as f:
        output_dict = pickle.load(f)

    with open(key_file, "rb") as f:
        answer_key = pickle.load(f)

    return output_dict, answer_key

def evaluate_metrics(output_dict, answer_key):
    common_keys = set(output_dict.keys()).intersection(set(answer_key.keys()))
    
    # Extract predictions and true values
    y_pred = [int(output_dict[k]) for k in common_keys]
    y_true = [int(answer_key[k]) for k in common_keys]

    # Calculate Accuracy
    correct_predictions = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    accuracy = correct_predictions / len(y_true) if y_true else 0

    # Calculate Precision and Recall
    label_counts = Counter(y_true)
    precision = 0
    recall = 0

    for label in label_counts:
        # True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = sum(1 for pred, true in zip(y_pred, y_true) if pred == label and true == label)
        fp = sum(1 for pred, true in zip(y_pred, y_true) if pred == label and true != label)
        fn = sum(1 for pred, true in zip(y_pred, y_true) if pred != label and true == label)

        # Calculate Precision and Recall for each label
        precision_label = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_label = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Average (Macro Precision/Recall)
        precision += precision_label
        recall += recall_label

    # Average over all labels
    precision /= len(label_counts) if label_counts else 0
    recall /= len(label_counts) if label_counts else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

if __name__ == "__main__":
    # File names
    output_file = "output_file_" + model + ".pkl"
    key_file = "answer_key_" + model + ".pkl"

    # Load the dictionaries
    output_dict, answer_key = load_dict_files(output_file, key_file)

    # Evaluate metrics
    metrics = evaluate_metrics(output_dict, answer_key)

    # Print results
    print("Model: " + model)
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
