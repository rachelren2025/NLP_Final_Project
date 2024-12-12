import json

from transformers import AutoTokenizer, AutoModelForMultipleChoice
from bert_score import score
import torch
import itertools
import math


def compute_similarity(holdings):
    """
    Compute pairwise similarity between holdings using BERTScore.
    Returns the geometric mean of all F1 scores.
    """
    f1_scores = []
    pairs = list(itertools.combinations(range(len(holdings)), 2))

    for i, j in pairs:
        _, _, F1 = score([holdings[i]], [holdings[j]], lang="en", verbose=False)
        f1_scores.append(F1.item())

    # Compute geometric mean of pairwise F1 scores
    geometric_mean_f1 = math.prod(f1_scores) ** (1 / len(f1_scores)) if f1_scores else 0.0
    return geometric_mean_f1


def newmetric(holdings, citing_prompt, logits, true_label):
    """
    Compute the new metric by combining BERT confidence, similarity scores, and reward/penalty.
    Args:
        holdings: List of 5 holdings.
        citing_prompt: The citing prompt as input.
        logits: BERT model logits for each holding.
        true_label: The correct label (0-4).
    Returns:
        metrics_dict: A dictionary containing correctness, similarity score, confidence, and newmetric.
    """
    # Step 1: Compute pairwise similarity geometric mean
    similarity_score = compute_similarity(holdings)

    # Step 2: Compute BERT confidence score
    confidence_scores = torch.softmax(logits, dim=-1)  # Normalize logits to probabilities
    predicted_label = logits.argmax(dim=-1).item()
    predicted_confidence = confidence_scores[0, predicted_label].item()

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
    # Load dataset and model
    from datasets import load_dataset

    ds = load_dataset("casehold/casehold", "all")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")

    results = {}  # Dictionary to store results for all examples

    # Iterate through the first few examples
    for idx, example in enumerate(ds["train"]):
        citing_prompt = example["citing_prompt"]
        holdings = [
            example["holding_0"],
            example["holding_1"],
            example["holding_2"],
            example["holding_3"],
            example["holding_4"],
        ]
        true_label = example["label"]
        example_id = example["example_id"]

        # Tokenize input for BERT
        inputs = tokenizer(
            [citing_prompt] * 5,  # Repeat the prompt for each holding
            holdings,  # Holdings to compare
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.unsqueeze(0) for key, value in inputs.items()}  # Add batch dimension

        # Forward pass through BERT
        outputs = model(**inputs)
        logits = outputs.logits


        # Compute new metric
        metrics = newmetric(holdings, citing_prompt, logits, true_label)

        # Store results in dictionary
        results[example_id] = metrics

        # Print for the first 3 examples (testing purposes)
        print(f"\nID: {example_id}")
        print(f"Metrics: {metrics}")
        if idx >= 2:  # Break after 3 examples for testing
            break

    with open('newmetric.txt', 'w') as f:
        json.dump(results, f, indent=4)
