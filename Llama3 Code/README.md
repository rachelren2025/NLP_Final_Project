# Requirements
- Python 3.7+
- Pandas library
- LLaMA model accessible via ollama CLI
- A CSV file (test.csv) containing legal case data 

# Installation

Install required Python libraries:

- pip install pandas

- Ensure the ollama CLI is installed and configured to run the LLaMA model.

# Usage
Place the CSV file (test.csv) in the project directory.

Run the script:

python llama_script.py

The script generates two dictionaries:

- output_dict: LLaMA's predictions for each case.
- answer_key: The actual correct answers for comparison. 

Example Output:

output_dict = {47821: "2", 47822: "1", 47823: "3"}  
answer_key = {47821: "2", 47822: "1", 47823: "3"}