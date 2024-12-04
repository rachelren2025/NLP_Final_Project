import pandas as pd
import subprocess
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

similarity_output = "cosine_similarity_scores.json"
responses_output = "response_for_cosine_similarity.json"
additional_metrics_output = "additional_metrics.json"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another pre-trained model

model = "llama3.2"
# Set to True to test only the first 3 entries
test = True

# Metric functions
def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def jaccard_similarity(vec1, vec2):
    intersection = np.minimum(vec1, vec2).sum()
    union = np.maximum(vec1, vec2).sum()
    return intersection / union if union != 0 else 0.0

def prompt_file(inp):
    # Send input to the process and get the output
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    try:
        command = """Instruction: You will receive a message representing the context or facts of a legal case.
        Your task is to generate the holding that is most relevant and aligns with the legal 
        principles or facts in the message.

        Format:

        Message: <Message> 

        Response: Generate the most relevant holding as a complete sentence or paragraph.
        """
        stdout, stderr = proc.communicate(input=command + inp, timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    print("STDOUT:", stdout)
    if proc.returncode == 0:
        print("Command executed successfully")
        return stdout
    else:
        print("Command failed with return code", proc.returncode)
        return "Failed"

def parse_data():
    similarity_scores = {}
    additional_metrics = {}
    output_dict = {}

    data = pd.read_csv('test.csv')
    json_data = eval(data.to_json(orient="records", indent=4))

    k = 0
    for i in json_data:
        if test and k == 3:
            break

        k += 1
        inp_str = 'Message: ' + i['0']
        print(inp_str)

        prompt_id = i['Unnamed: 0']
        out = prompt_file(inp_str).strip()
        output_dict[prompt_id] = out
        print(out)

        # Retrieve the correct holding text
        correct_option = int(i['11'])
        correct_holding = i[str(correct_option)]
        print(f"Correct Holding for Prompt ID {prompt_id}: {correct_holding}")

        # Calculate embeddings
        model_embedding = embedding_model.encode([out])[0]
        key_embedding = embedding_model.encode([correct_holding])[0]

        # Cosine Similarity
        similarity = cosine_similarity([model_embedding], [key_embedding])[0][0]
        similarity_scores[prompt_id] = float(similarity)  # Convert to Python-native float

        # Additional metrics
        dot = dot_product(model_embedding, key_embedding)
        euclidean = euclidean_distance(model_embedding, key_embedding)
        manhattan = manhattan_distance(model_embedding, key_embedding)
        jaccard = jaccard_similarity(model_embedding, key_embedding)

        additional_metrics[prompt_id] = {
            "dot_product": float(dot),  # Convert to Python-native float
            "euclidean_distance": float(euclidean),  # Convert to Python-native float
            "manhattan_distance": float(manhattan),  # Convert to Python-native float
            "jaccard_similarity": float(jaccard),  # Convert to Python-native float
        }

        print(f"Cosine Similarity for Prompt ID {prompt_id}: {similarity:.4f}")
        print(f"Dot Product for Prompt ID {prompt_id}: {dot:.4f}")
        print(f"Euclidean Distance for Prompt ID {prompt_id}: {euclidean:.4f}")
        print(f"Manhattan Distance for Prompt ID {prompt_id}: {manhattan:.4f}")
        print(f"Jaccard Similarity for Prompt ID {prompt_id}: {jaccard:.4f}")

    # Save results to files
    with open(similarity_output, "w") as f:
        json.dump(similarity_scores, f, indent=4)

    with open(responses_output, "w") as f:
        json.dump(output_dict, f, indent=4)

    with open(additional_metrics_output, "w") as f:
        json.dump(additional_metrics, f, indent=4)

    print("Results dumped")


if __name__ == "__main__":
    parse_data()
