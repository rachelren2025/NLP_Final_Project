import json
import math


def compute_geometric_mean(scores):
    """
    Compute the geometric mean of a list of scores.
    Args:
        scores: List of pairwise scores.
    Returns:
        geometric_mean: Geometric mean of the scores.
    """
    return math.prod(scores) ** (1 / len(scores)) if scores else 0.0


def compute_standard_deviation(scores):
    """
    Compute the variance of a list of scores.
    Args:
        scores: List of pairwise scores.
    Returns:
        variance: Variance of the scores.
    """
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return math.sqrt(variance)


if __name__ == "__main__":
    # Load the pairwise scores JSON file
    with open("pairwise_scores.json", "r") as f:
        pairwise_scores = json.load(f)

    difficulty_results = {}  # Dictionary to store difficulty metrics

    # Iterate through each example and calculate difficulty metrics
    for example_id, scores in pairwise_scores.items():
        geometric_mean = compute_geometric_mean(scores)
        standard_deviation = compute_standard_deviation(scores)

        # Store results as {id: {geometric_mean, variance, standard_deviation}}
        difficulty_results[example_id] = {
            "geometric_mean": round(geometric_mean, 4),
            "standard_deviation": round(standard_deviation, 4)
        }

        # Print progress for verification
        print(f"Processed Example ID: {example_id}, "
              f"Geometric Mean: {geometric_mean}, "
              f"Standard Deviation: {standard_deviation}")

    # Save the results to a new JSON file
    with open("difficulty_metrics.json", "w") as f:
        json.dump(difficulty_results, f, indent=4)

    print("\nDifficulty metrics saved to 'difficulty_metrics.json'.")
