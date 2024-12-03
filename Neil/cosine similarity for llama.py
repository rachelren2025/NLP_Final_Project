import pandas as pd
import subprocess
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

similarity_output = "cosine_similarity_scores.json"
responses_output = "response_for_cosine_similarity.json"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another pre-trained model

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
        # Define the prompt command
        command = """Instruction: You will receive a message representing the context or facts of a legal case.
        Your task is to generate the holding that is most relevant and aligns with the legal 
        principles or facts in the message.

    Format:

    Message: <Message> 

    Response: Generate the most relevant holding as a complete sentence or paragraph.
    
    For example:
    
    Message: even though the store benefitted from the advertising, the court found this did not rise to the level of consideration. Id. In State v. Socony Mobil Oil Co., the Court of Civil Appeals contrasted Brice with Cole and found no consideration where a filling station paid for bingo cards but gave them away free to any and all persons who came to their stations to request them, and a local TV station broadcast games in which bingo cards were used with winners being awarded cash prizes. 386 S.W.2d 169 (Tex.Civ.App.-San Antonio 1964). Some jurisdictions outside of the State of Texas have held that requiring a person to actually go to the location of the sweepstakes sponsor in order to participate constitutes consideration. See Lucky Calendar Co. v. Cohen, 19 N.J. 399, 117 A.2d 487, 496 (1955)(<HOLDING>); Knox Indus. Corp. v. State ex rel. Scanland,
    
    Response: holding that murder committed by customer was not foreseeable result of excessive sale of alcohol to customer
    """

        stdout, stderr = proc.communicate(input=command + inp, timeout=30)

    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    print("STDOUT:", stdout)
    #print("STDERR:", stderr)

    # Check if the process ended successfully
    if proc.returncode == 0:
        print("Command executed successfully")
        return stdout
    else:
        print("Command failed with return code", proc.returncode)
        return "Failed"


def parse_data():
    similarity_scores = {}
    output_dict = {}

    data = pd.read_csv('test.csv')
    json_data = eval(data.to_json(orient="records", indent=4))

    k = 0
    for i in json_data:
        if test and k == 3:
            break

        k += 1


        # Construct the input string for the model
        inp_str = (
                'Message: ' + i['0']
        )

        print(inp_str)

        prompt_id = i['Unnamed: 0']
        out = prompt_file(inp_str).strip()

        output_dict[prompt_id] = out

        # Retrieve the correct holding text based on column `11`
        correct_option = int(i['11'])  # Convert to integer
        correct_holding = i[str(correct_option)]  # Map the option to the correct holding column

        print(f"Correct Holding for Prompt ID {prompt_id}: {correct_holding}")

        # Calculate embeddings and cosine similarity
        model_embedding = embedding_model.encode([out])
        key_embedding = embedding_model.encode([correct_holding])
        similarity = cosine_similarity(model_embedding, key_embedding)

        # Store and print the similarity score
        similarity_scores[prompt_id] = similarity[0][0]
        similarity_scores = {k: float(v) for k, v in similarity_scores.items()}  # Convert to float for json
        print(f"Cosine Similarity for Prompt ID {prompt_id}: {similarity[0][0]}")

    # Print all similarity scores
    print("All Similarity Scores:", similarity_scores)
    print(output_dict)

    with open(similarity_output, "w") as f:
        json.dump(similarity_scores, f, indent=4)

    with open(responses_output, "w") as f:
        json.dump(output_dict, f, indent=4)

    print("results dumped")



if __name__ == "__main__":
    parse_data()
