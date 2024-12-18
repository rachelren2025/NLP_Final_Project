import json
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

model = "phi3"
test_results = False


def load_dict_files(output_file, answer_key):
    with open(output_file, "rb") as f:
        output_dict = json.load(f)

    with open(answer_key, "rb") as f:
        answer_key = json.load(f)

    return output_dict, answer_key

def clean_results(model_results):
    # Use regex to clean data
    cleaned_data = {}

    # Regex patterns
    only_number_pattern = r"^(0|1|2|3|4)$"
    response_pattern = r"Response:\s*(0|1|2|3|4)"
    beginning_pattern = r"^\"?(0|1|2|3|4):"

    for prompt_id, output in model_results.items():
        extracted_number = None

        # Check for "<number>"
        only_number_match = re.match(only_number_pattern, output.strip())
        if only_number_match:
            extracted_number = int(only_number_match.group(1))
        else:
            # Check for "Response: <number>"
            response_match = re.search(response_pattern, output)
            if response_match:
                extracted_number = int(response_match.group(1))
            else:
                # Check for "<number>: "
                beginning_of_text_match = re.match(beginning_pattern, output)
                if beginning_of_text_match:
                    extracted_number = int(beginning_of_text_match.group(1))

        # Default to 9 if no valid number is extracted
        cleaned_data[prompt_id] = extracted_number if extracted_number is not None else "9"

    return cleaned_data

# Metrics
def compute_accuracy(model_results, answer_key):
    correct = 0
    total = len(model_results)

    for prompt_id, answer in answer_key.items():
        if prompt_id in model_results and str(model_results[prompt_id]) == str(answer):
            correct += 1

    accuracy = (correct / total) if total > 0 else 0
    return accuracy

def compute_mean_weighted_precision_recall(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    return precision, recall

def compute_weighted_accuracy(answer_key, system_output):
    true_counts = Counter(answer_key.values())  # Total occurrences of each class
    weights = {c: count for c, count in true_counts.items()}  # Use class frequencies as weights

    correct = Counter()
    for qid in answer_key:
        if answer_key[qid] == system_output.get(qid, None):
            correct[answer_key[qid]] += 1

    weighted_sum = sum(weights[c] * correct[c] for c in true_counts)
    total_weighted = sum(weights[c] * true_counts[c] for c in true_counts)

    return weighted_sum / total_weighted

def compute_f1(precision, recall):
    if precision + recall == 0:  # Avoid division by zero
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_macro_f1(y_true, y_pred):
    """
    Compute the Macro F1 score (average F1 across all classes).
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def compute_micro_f1(y_true, y_pred):
    """
    Compute the Micro F1 score (aggregated metrics across all classes).
    """
    return f1_score(y_true, y_pred, average='micro', zero_division=0)

if __name__ == "__main__":
    # File names
    if test_results:
        model_results_filename = "results/output_file_" + model + ".json"
    else:
        model_results_filename = "results/output_file_" + model + ".json"
    answer_key_filename = "results/answer_key.json"

    # Load files
    model_results, answer_key = load_dict_files(model_results_filename, answer_key_filename)

    # Clean results
    model_results = clean_results(model_results)  # clean data

    # Prepare Data for weighted_precision, weighted_recall, macro_f1, micro_f1
    y_true = []
    y_pred = []
    for prompt_id in answer_key.keys():
        if prompt_id in model_results:
            y_true.append(answer_key[prompt_id])
            y_pred.append(int(model_results[prompt_id]))
    
    # Evaluate metrics
    accuracy = compute_accuracy(model_results, answer_key)
    weighted_accuracy = compute_weighted_accuracy(model_results, answer_key)
    weighted_precision, weighted_recall = compute_mean_weighted_precision_recall(y_true, y_pred)
    f1 = compute_f1(weighted_precision, weighted_recall)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    micro_f1 = compute_micro_f1(y_true, y_pred)

    # Print results
    print("Model: " + model)
    print("Evaluation Metrics:")
    print(f"accuracy: {accuracy:.4f}")
    print(f"weighted accuracy: {weighted_accuracy:.4f}")
    print(f"weighted precision: {weighted_precision:.4f}")
    print(f"weighted recall: {weighted_recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
