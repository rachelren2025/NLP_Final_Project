import pandas as pd
import subprocess
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another pre-trained model

output_dict = {}
answer_key = {}
similarity_scores = {}
model = "llama3.2"
# Set to True to test only the first 3 entries
test = True


def prompt_file(inp):
    # Send input to the process and get the output
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,  # Enables sending input
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard errors
        text=True,  # Inputs and outputs are handled as text
        encoding="utf-8",  # Ensure the process uses utf-8 encoding
        errors="replace"  # Replace invalid characters
    )

    try:
        # Define the prompt command (modify as per your requirement)
        command = """Instruction: You will receive a message representing the context or facts of a legal case, followed by four 
        possible legal holdings. Your task is to select the holding that is most relevant and aligns with the legal 
        principles or facts in the message. Output only the number of the most relevant holding.

        Format:

        Message: <Message> 1: <Option 1> 2: <Option 2> 3: <Option 3> 4: <Option 4> 
        Response: 
        """

        stdout, stderr = proc.communicate(input=command + inp, timeout=30)

    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    print("STDOUT:", stdout)
    print("STDERR:", stderr)

    # Check if the process ended successfully
    if proc.returncode == 0:
        print("Command executed successfully")
        return stdout
    else:
        print("Command failed with return code", proc.returncode)
        return "Failed"


def parse_data():
    data = pd.read_csv('test.csv')
    json_data = eval(data.to_json(orient="records", indent=4))

    k = 0
    for i in json_data:
        if test and k == 3:
            break

        k += 1

        inp_str = (
            'Message: <' + i['0'] + '> 1: <' + i['1'] + '> 2: <' + i['2'] + '> 3: <' +
            i['3'] + '> 4: <' + i['4'] + '>'
        )

        print("Input String:", inp_str)

        prompt_id = i['Unnamed: 0']
        out = prompt_file(inp_str).strip()

        output_dict[prompt_id] = out
        answer_key[prompt_id] = i['11']

        # Calculate cosine similarity between model output and answer key
        model_embedding = embedding_model.encode([out])
        key_embedding = embedding_model.encode([i['11']])
        similarity = cosine_similarity(model_embedding, key_embedding)

        # Store and print the similarity score
        similarity_scores[prompt_id] = similarity[0][0]
        print(f"Cosine Similarity for Prompt ID {prompt_id}: {similarity[0][0]}")

    # Print all similarity scores
    print("All Similarity Scores:", similarity_scores)


if __name__ == "__main__":
    parse_data()
