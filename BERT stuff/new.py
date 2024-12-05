from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch  # May need for GPU - ask GPT

"""Load the dataset"""
from datasets import load_dataset

ds = load_dataset("casehold/casehold", "all")

"""Load the tokenizer and model"""
# Loads a tokenizer for BERT. It converts text into token IDs for input into the model.#

# Load LegalBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForMultipleChoice.from_pretrained("nlpaueb/legal-bert-base-uncased")

print("LegalBERT loaded successfully!")

# Get the first example from the file - will need to adjust for all 20k whatever
first_example = ds["train"][0]
citing_prompt = first_example["citing_prompt"]
holdings = [
    first_example["holding_0"],
    first_example["holding_1"],
    first_example["holding_2"],
    first_example["holding_3"],
    first_example["holding_4"],
]

# Tokenize the input
inputs = tokenizer(
    [citing_prompt] * 5,  # Repeat the citing_prompt for each holding -
    holdings,  # Each holding as the paired text
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)

# Reshape inputs for BertForMultipleChoice
"""Adds a batch dimension to the inputs. The model expects batched inputs, even if it’s just one example.
After this, each tensor in inputs has a shape like [1, 5, 512] (batch size = 1, 5 choices, 512 tokens).
Will need to adjust for the total dataset."""

inputs = {
    key: value.unsqueeze(0) for key, value in inputs.items()
    # Adds a batch dimension to each tensor
}

# Forward pass through BERT
outputs = model(**inputs)
logits = outputs.logits  # Returns logits (unnormalized scores) for each of the 5 choices.

# Interpret logits - Finds the index of the maximum logit along the choice dimension (best answer).
predicted_label = logits.argmax(dim=-1).item()

# Print results
print(f"\nCiting Prompt: {citing_prompt}")
print(f"\nHoldings:")

for idx, holding in enumerate(holdings):
    print(f"  {idx}: {holding}")

print(f"\nPredicted Label: {predicted_label}")
# print(f"Predicted Holding: {holdings[predicted_label]}")
true_label = first_example["label"]
print(f"Actual label: {true_label}")
