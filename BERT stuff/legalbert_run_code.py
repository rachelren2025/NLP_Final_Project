import json
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch
from torch.nn.functional import softmax
from datasets import load_dataset

# Set a fixed random seed
torch.manual_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
ds = load_dataset("casehold/casehold", "all")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("zlucia/legalbert")
model = AutoModelForMultipleChoice.from_pretrained("zlucia/legalbert").to(device)
print("LegalBERT loaded successfully!")

model.eval()  # Set model to evaluation mode

# List to store the results
results = []

total_count = len(ds["train"])

# Process the examples
for i in range(total_count):  # Loop through entire dataset
    try:
        example = ds["train"][i]  # Access by index

        # Access the required fields
        citing_prompt = example.get("citing_prompt", "")
        holdings = [
            example.get("holding_0", ""),
            example.get("holding_1", ""),
            example.get("holding_2", ""),
            example.get("holding_3", ""),
            example.get("holding_4", ""),
        ]

        # Tokenize the input
        encoded_inputs = tokenizer(
            [citing_prompt] * 5,  # Repeat the citing_prompt for each holding
            holdings,  # Each holding as the paired text
            truncation=True,
            padding="max_length",
            max_length=256,  # Reduce max length to reduce memory
            return_tensors="pt",
        )

        # Reshape inputs for multiple-choice format
        inputs = {
            key: value.view(1, 5, -1).to(device)  # Reshape to [1 (batch size), 5 (num_choices), seq_length]
            for key, value in encoded_inputs.items()
        }

        # Forward pass through BERT with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

        logits = outputs.logits  # Returns logits (unnormalized scores) for each of the 5 choices

        # Apply softmax to logits to get probabilities
        probabilities = softmax(logits, dim=-1).squeeze().tolist()

        # Interpret logits to find the predicted label
        predicted_label = logits.argmax(dim=-1).item()

        # Save the results
        result = {
            "example_index": i,
            "citing_prompt": citing_prompt,
            "holdings": holdings,
            "probabilities": probabilities,
            "predicted_label": predicted_label,
            "correct_label": example["label"],
        }
        results.append(result)

        # Clear GPU cache
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Skipping example {i} due to memory error.")
            torch.cuda.empty_cache()

    # Print progress
    if i % 100 == 0:
        print(f"Processed {i + 1}/{total_count} examples...")

# Save the results to a JSON file
output_file = "casehold_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
