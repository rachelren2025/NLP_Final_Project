# Requirements
- Python 3.7+
- Pandas library
- LLaMA model accessible via ollama CLI
- A CSV file (test.csv) containing legal case data 

# Installation

Install required Python libraries:

- pip install pandas

- Ensure the ollama CLI is installed and configured to run the LLaMA model.

# Conditions
`model` - input model name here
`test` variable set true if testing with inputs, false if running entire data set

# Usage
Place the CSV file (test.csv) in the project directory.

Run the script:

python ollama_script.py

The script generates two dictionaries:

- output_dict: model's predictions for each case.
- answer_key: The actual correct answers for comparison. 

Example Output:

"output_file_<model_name>.pkl" = {47821: "2", 47822: "1", 47823: "3"}  
"answer_key_<model_name>.pkl" = {47821: "2", 47822: "4", 47823: "1"}