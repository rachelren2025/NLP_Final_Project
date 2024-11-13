from transformers import pipeline

# Load the model pipeline for fill-mask
pipe = pipeline("fill-mask", model="casehold/custom-legalbert")

# Path to your dataset
file_path = "test.csv"

# Open and process the file
with open(file_path, "r") as f:
    lines = f.readlines()

# Iterate through each line and process it
for i, line in enumerate(lines):
    line = line.strip()  # Remove extra spaces or newline characters
    if "[MASK]" in line:  # Ensure the line has a [MASK] token
        print(f"Processing line {i + 1}: {line}")
        predictions = pipe(line)
        print(f"Predictions for line {i + 1}:")
        for pred in predictions:
            print(f"  {pred['sequence']} (Score: {pred['score']:.4f})")
    else:
        print(f"Skipping line {i + 1}: No [MASK] token found.")
