import csv
import json

def parse_line(data_line):
    """
    Parses a single line of input into structured components.
    Args:
        data_line (str): A single line of raw input.
    Returns:
        dict: Parsed components of the input line.
    """
    # Step 1: Split by comma (handle quoted text properly using csv module)
    csv_reader = csv.reader([data_line], quotechar='"', delimiter=',', skipinitialspace=True)
    parsed_line = list(csv_reader)[0]

    # Step 2: Extract components
    parsed_data = {
        "id": parsed_line[0],  # ID
        "text": parsed_line[1],  # Main text
        "options": parsed_line[2:7],  # Options (next 5 fields)
        "scores": [float(score) for score in parsed_line[7:12]],  # Convert scores to floats
        "correct_label": int(parsed_line[12])  # Correct label
    }

    return parsed_data


dev_file = "../Casehold_code/data/dev.csv"

p = []
with open(dev_file, 'r') as f:
    k = f.readlines()
for i in k[1:]:
    p.append(parse_line(i))

with open('parsed_dev_file.json', 'w') as g:
    json.dump(p, g, indent=4)

