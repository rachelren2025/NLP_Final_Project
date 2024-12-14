import json

def split_dataset(data, output_file):
    """
    Split the dataset into four quartiles (q1, q2, q3, q4) based on the values.
    Args:
        data: Dictionary of questions with their metrics.
        output_file: Name of the output file.
    """
    # Sort questions by the values (scores)
    sorted_data = sorted(data.items(), key=lambda x: x[1])

    # Calculate quartile sizes
    total = len(sorted_data)
    q_size = total // 4

    # Split into quartiles
    q1 = sorted_data[:q_size]  # Bottom 25%
    q2 = sorted_data[q_size:2*q_size]  # 25% - 50%
    q3 = sorted_data[2*q_size:3*q_size]  # 50% - 75%
    q4 = sorted_data[3*q_size:]  # Top 25%

    # Create output dictionary
    output = {}
    for prompt_id, _ in q1:
        output[prompt_id] = "q1"  # Bottom 25%
    for prompt_id, _ in q2:
        output[prompt_id] = "q2"
    for prompt_id, _ in q3:
        output[prompt_id] = "q3"
    for prompt_id, _ in q4:
        output[prompt_id] = "q4"  # Top 25%

    # Sort the output dictionary by keys (prompt IDs)
    sorted_output = dict(sorted(output.items(), key=lambda x: int(x[0])))

    # Write the results to the output file
    with open(output_file, "w") as f:
        json.dump(sorted_output, f, indent=4)

    print(f"Results saved to '{output_file}'")


if __name__ == "__main__":
    # Load the metrics JSON file
    with open("bertscore_results.json", "r") as f:
        data = json.load(f)

    # Split and save into quartiles
    split_dataset(data, output_file="split_datasets/GM_sorted_questions.json")
