import json
import numpy as np


def compute_new_metric(difficulty_file, confidence_file):
    """
    Compute the new metric S_i = D_i * C_i * V_i for each question.

    Args:
        difficulty_file (str): Path to the file containing question difficulty levels.
        confidence_file (str): Path to the file containing confidence scores and correctness.
        output_file (str): Path to save the final results.
    """

    # Define difficulty multipliers for each quarter
    difficulty_multipliers = {
        "q1": 0.25,  # Easiest questions
        "q2": 0.5,
        "q3": 0.75,
        "q4": 1.0  # Hardest questions
    }

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

    dev_file_results = []
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
        output_dict["new_metric"] = max(r) * output_dict["difficulty"] * (
            1 if output_dict["bert_choice"] == output_dict["correct_label"] else -1)
        results.append(output_dict)

        i += 1

    with open('parsed_dev_file_with_newmetric.json', 'w') as out:
        json.dump(results, out, indent=4)


def split_quartiles_by_bertscore(file_path):
    """
    Split the data into quartiles based on difficulty (BERTScore)
    and compute the average new_metric for each quartile.

    Args:
        file_path (str): Path to the input JSON file.

    Returns:
        dict: Average new_metric for each quartile.
    """
    # Step 1: Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Step 2: Sort data by difficulty (BERTScore) in ascending order
    sorted_data = sorted(data, key=lambda x: x['difficulty'])

    # Step 3: Split into 4 quartiles
    q1, q2, q3, q4 = np.array_split(sorted_data, 4)

    # Step 4: Compute average new_metric for each quartile
    def compute_avg_newmetric(quartile):
        return round(np.mean([entry['new_metric'] for entry in quartile]), 4)

    quartile_averages = {
        "Q1 (Easiest)": compute_avg_newmetric(q1),
        "Q2": compute_avg_newmetric(q2),
        "Q3": compute_avg_newmetric(q3),
        "Q4 (Hardest)": compute_avg_newmetric(q4)
    }

    return quartile_averages



if __name__ == "__main__":
    # Input file paths
    difficulty_file = "./bertscore_results.json"  # File 1: {id: "q1", "q2", ...}
    confidence_file = "../Casehold_code/output/bert-double/probabilities.csv"  # Line 1 corresponds to first id
    dev_file = "./parsed_dev_file.json"
    output_file = "new_metric_results.json"

    #compute_new_metric(difficulty_file, confidence_file)
    split_quartiles_by_bertscore('parsed_dev_file_with_newmetric.json')

    quartile_results = split_quartiles_by_bertscore('parsed_dev_file_with_newmetric.json')
    # Print the results
    for quartile, avg in quartile_results.items():
        print(f"{quartile}: {avg}")


