import json
import math
from bert_score import score
import itertools
from datasets import load_dataset

def compute_similarity(holdings):
    """
    Compute pairwise similarity between holdings using BERTScore.
    Returns the geometric mean of all F1 scores.
    Args:
        holdings: List of answer choices for a question.
    """
    f1_scores = []
    pairs = list(itertools.combinations(range(len(holdings)), 2))

    # Compute BERTScore F1 for each pair of holdings
    for i, j in pairs:
        _, _, F1 = score([holdings[i]], [holdings[j]], lang="en", verbose=False)
        f1_scores.append(F1.item())

    # Compute geometric mean of pairwise F1 scores
    geometric_mean_f1 = math.prod(f1_scores) ** (1 / len(f1_scores)) if f1_scores else 0.0
    return round(geometric_mean_f1, 4)  # Round for cleaner output


if __name__ == "__main__":
    # Load the CaseHold dataset
    ds = load_dataset("casehold/casehold", "all")

    results = {}  # Dictionary to store BERTScore results

    # Iterate through the first N examples (you can remove [:N] to run on the full dataset)
    for example in ds["train"]:
        example_id = example["example_id"]
        holdings = [
            example["holding_0"],
            example["holding_1"],
            example["holding_2"],
            example["holding_3"],
            example["holding_4"]
        ]

        # Compute BERTScore similarity
        bertscore = compute_similarity(holdings)

        # Store result as {id: score}
        results[example_id] = bertscore

        # Print progress for verification
        print(f"Processed Example ID: {example_id}, BERTScore: {bertscore}")

        # break

    # Save the results to a JSON file
    with open("bertscore_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nBERTScore results saved to 'bertscore_results.json'.")
