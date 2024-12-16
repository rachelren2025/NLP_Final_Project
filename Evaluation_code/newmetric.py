import json
import numpy as np


def combine_files_before_newmetric(difficulty_file, confidence_file, dev_file):
    # Step 1: Load input files
    with open(difficulty_file, "r") as f:
        difficulty_data = json.load(f)  # Format: {id: "q1", ...}

    with open(confidence_file, "r") as f:
        confidence_data = f.readlines()

    softmax_results = []  # Store softmax probabilities for all examples

    for i in confidence_data:
        # Convert list elements to float
        i = np.array([float(x) for x in i.split(',')])

        # Compute softmax
        exp_values = np.exp(i - np.max(i))  # Numerical stability: subtract max(i)
        probabilities = exp_values / np.sum(exp_values)
        softmax_results.append(probabilities.tolist())  # Append softmax probabilities

    with open(dev_file, 'r') as d:
        dev_file_results = json.load(d)

    i = 0
    results = []
    for key, val in difficulty_data.items():
        r = softmax_results[i]
        bert_output = r.index(max(r))
        output_dict = dev_file_results[i]

        output_dict["softmax_scores"] = r
        output_dict["bert_choice"] = bert_output
        output_dict["correctness"] = output_dict["bert_choice"] == output_dict["correct_label"]
        output_dict["difficulty"] = difficulty_data[key]

        results.append(output_dict)

        i += 1

    with open('parsed_dev_file_with_bert_score.json', 'w') as out:
        json.dump(results, out, indent=4)


def calculate_new_metric(input_file, output_file):
    """
    Split questions into quartiles based on difficulty, assign weights, compute new metric,
    and print the average new_metric for each quartile.

    Args:
        input_file (str): Path to the JSON file containing question data.
        output_file (str): Path to save the updated JSON file with new metric.
    """
    # Step 1: Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Step 2: Sort data by difficulty in ascending order
    sorted_data = sorted(data, key=lambda x: x['difficulty'])

    # Step 3: Split into 4 quartiles
    quartiles = np.array_split(sorted_data, 4)

    # Step 4: Define weights for each quartile
    difficulty_weights = [0.25, 0.5, 0.75, 1.0]  # Q1 -> Q4

    # Step 5: Compute the new metric and track averages
    results = []
    quartile_sums = [0, 0, 0, 0]  # To store sum of new_metric for each quartile
    quartile_counts = [0, 0, 0, 0]  # To count the number of items in each quartile

    for i, quartile in enumerate(quartiles):
        weight = difficulty_weights[i]  # Weight for this quartile

        for question in quartile:
            confidence = max(question["softmax_scores"])  # Highest softmax score (C_i)
            correctness = 1 if question["bert_choice"] == question["correct_label"] else -1  # Correctness (V_i)

            # Compute new metric: S_i = D_i * C_i * V_i
            new_metric = round(weight * confidence * correctness, 4)

            # Update question with the new metric
            question.update({
                "difficulty_weight": weight,
                "new_metric": new_metric
            })
            results.append(question)

            # Track sums and counts for averages
            quartile_sums[i] += new_metric
            quartile_counts[i] += 1

    # Step 6: Save updated data to output file
    with open(output_file, 'w') as out:
        json.dump(results, out, indent=4)

    # Step 7: Print average new_metric for each quartile
    print("Average new_metric per quartile:")
    for i, (total, count) in enumerate(zip(quartile_sums, quartile_counts)):
        average = round(total / count, 4) if count > 0 else 0.0
        print(f"Q{i + 1} (Weight: {difficulty_weights[i]}): {average}")

    print(f"Results with new metric saved to '{output_file}'")


if __name__ == "__main__":
    # Input file paths
    difficulty_file = "./bertscore_results.json"  # File 1: {id: "q1", "q2", ...}
    confidence_file = "../Casehold_code/output/bert-double/probabilities.csv"  # Line 1 corresponds to first id
    dev_file = "./parsed_dev_file.json"

    combine_files_before_newmetric(difficulty_file, confidence_file, dev_file)

    output_file = "new_metric_results.json"
    calculate_new_metric("parsed_dev_file_with_bert_score.json", output_file)
