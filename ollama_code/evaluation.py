import json
import re
from collections import Counter
from sklearn.metrics import precision_score

model = "llama3.2"


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


def compute_mean_weighted_precision(model_results, answer_key):
    classes = {0, 1, 2, 3, 4, 9}
    # Convert predicted labels to integers
    y_true = []
    y_pred = []
    for prompt_id in answer_key.keys():
        if prompt_id in model_results:
            y_true.append(answer_key[prompt_id])
            y_pred.append(int(model_results[prompt_id]))  # Convert string to int

    # Calculate the weighted mean precision
    precision = precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
    return precision


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


if __name__ == "__main__":
    # File names
    # Seems to be that \\ doesn't work for mac - use /
    #model_results_filename = "results\\output_file_" + model + ".json"
    model_results_filename = "results/output_file_llama3.2.json"
    #answer_key_filename = "results\\answer_key.json"
    answer_key_filename = "results/answer_key.json"

    # Load files
    model_results, answer_key = load_dict_files(model_results_filename, answer_key_filename)

    # Clean results
    model_results = clean_results(model_results)  # clean data

    # Evaluate metrics
    #accuracy = compute_accuracy(model_results, answer_key)

    weighted = weighted_accuracy(model_results, answer_key)

    # Print results
    print("Model: " + model)
    print("Evaluation Metrics:")
    print(f"weighted: {weighted:.4f}")
