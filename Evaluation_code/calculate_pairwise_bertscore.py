import json
import itertools
from bert_score import score
import torch
from datasets import load_dataset


def compute_pairwise_scores(holdings, device):
    """
    Compute pairwise BERTScore F1 for all combinations of holdings.
    Args:
        holdings: List of answer choices for a question.
        device: The device to use (CPU or CUDA).
    Returns:
        pairwise_scores: List of BERTScore F1 scores for each pair.
    """
    pairwise_scores = []
    pairs = list(itertools.combinations(range(len(holdings)), 2))

    # Compute BERTScore F1 for each pair
    for i, j in pairs:
        _, _, F1 = score([holdings[i]], [holdings[j]], lang="en", verbose=False, device=device)
        pairwise_scores.append(F1.item())
    
    return pairwise_scores


if __name__ == "__main__":
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the CaseHold dataset
    ds = load_dataset("casehold/casehold", "all")

    results = {}  # Dictionary to store pairwise BERTScore results

    # Iterate through the examples in the validation set
    for example in ds["validation"]:
        example_id = example["example_id"]
        holdings = [
            example["holding_0"],
            example["holding_1"],
            example["holding_2"],
            example["holding_3"],
            example["holding_4"]
        ]

        # Compute pairwise BERTScore for the holdings
        pairwise_scores = compute_pairwise_scores(holdings, device=device)

        # Store result as {id: [scores]}
        results[example_id] = pairwise_scores

        # Print progress for verification
        print(f"Processed Example ID: {example_id}, Pairwise Scores: {pairwise_scores}")

        # Uncomment to limit to a subset of examples for testing
        # if example_id == 42508 + 30:
        #     break

    # Save the results to a JSON file
    with open("pairwise_scores.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nPairwise scores saved to 'pairwise_scores.json'.")
