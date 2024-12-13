import json
import math
from bert_score import score
import itertools
import torch

def compute_similarity(holdings):
    """
    Compute pairwise similarity between holdings using BERTScore.
    Returns the geometric mean of all F1 scores.
    Args:
        holdings: List of answer choices for a question.
    """
    f1_scores = []
    pairs = list(itertools.combinations(range(len(holdings)), 2))

    # Compute BERTScore F1 for each pair
    for i, j in pairs:
        _, _, F1 = score([holdings[i]], [holdings[j]], lang="en", verbose=False)
        f1_scores.append(F1.item())

    # Compute geometric mean of pairwise F1 scores
    geometric_mean_f1 = math.prod(f1_scores) ** (1 / len(f1_scores)) if f1_scores else 0.0
    return geometric_mean_f1


def newmetric_with_precomputed_confidence(holdings, confidence_scores, true_label):
    """
    Compute the new metric using precomputed confidence scores and dynamically computed similarity.
    Args:
        holdings: List of 5 holdings (answer choices).
        confidence_scores: List of softmax probabilities for each choice.
        true_label: The correct label (0-4).
    Returns:
        metrics_dict: A dictionary containing correctness, similarity score, confidence, and newmetric.
    """
    # Step 1: Compute pairwise similarity geometric mean
    similarity_score = compute_similarity(holdings)

    # Step 2: Find the predicted label and confidence
    predicted_label = confidence_scores.index(max(confidence_scores))  # Index of the highest probability
    predicted_confidence = max(confidence_scores)

    # Step 3: Correctness
    correct = predicted_label == true_label

    # Step 4: Combine confidence, similarity, and correctness into the final metric
    reward_penalty = 1 if correct else -1
    new_metric = predicted_confidence * similarity_score * reward_penalty

    # Step 5: Return metrics as a dictionary
    metrics_dict = {
        'correct': correct,
        'similarity_score': round(similarity_score, 4),
        'confidence': round(predicted_confidence, 4),
        'newmetric': round(new_metric, 4)
    }
    return metrics_dict


# Example Usage
if __name__ == "__main__":
    # Load precomputed data (example file: precomputed_confidence.json)
    with open("zlucia_legalbert_casehold_results.json", "r") as f:
        data = json.load(f)

    results = {}  # Dictionary to store results for all examples

    # Iterate through the examples
    for example in data:
        example_id = example["example_index"]
        holdings = example["holdings"]  # List of 5 answer choices
        confidence_scores = example["probabilities"]  # Precomputed softmax probabilities
        true_label = example["correct_label"]

        # Compute the new metric
        metrics = newmetric_with_precomputed_confidence(holdings, confidence_scores, true_label)

        # Store results
        results[example_id] = metrics

        # Print results for verification
        print(f"\nID: {example_id}")
        print(f"Metrics: {metrics}")

        # break

    # Save results to a file
    with open("newmetric_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to 'newmetric_results.json'")
