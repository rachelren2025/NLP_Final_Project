import pandas as pd
import subprocess
import json
import time
import pickle

output_dict = {}
answer_key = {}
model = "llama3.2"
# set to True to pass the first 10 into the model
test = False

def prompt_file(inp):
    # Send input to the process and get the output

    # Start the process
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
        command = """

        Instruction: You will receive a message representing the context or facts of a legal case, followed by four 
        possible legal holdings. Your task is to select the holding that is most relevant and aligns with the legal 
        principles or facts in the message. Output only one number and nothing else at all.

        Format:

        Message: <Message> 1: <Option 1> 2: <Option 2> 3: <Option 3> 4: <Option 4> Response: Return the number (1, 2, 
        3, or 4) corresponding to the holding that is the most contextually relevant to the message.

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
        if test == True and k == 10:
            break

        k += 1

        inp_str = 'Message: <' + i['0'] + '> 1: <' + i['1'] + '> 2: <' + i['2'] + '> 3: <' + i['3'] + '> 4: <' + i[
            '4'] + '>'

        print(inp_str)

        prompt_id = i['Unnamed: 0']
        out = prompt_file(inp_str)
        output_dict[prompt_id] = out.strip()
        answer_key[prompt_id] = i['11']

        print("\n\n")

def mark_invalid_answers(ouput_dict):   #set all invalid answers to 9
    for prompt_id, output_answer in output_dict.items():
        try:
            num = int(output_answer)
            if num < 0 or num > 4:
                output_dict[prompt_id] = 9
        except ValueError:
            output_dict[prompt_id] = 9

def save_output(output_dict , answer_key):
    with open("output_file_"+ model + ".pkl", "wb") as f:
        pickle.dump(output_dict, f)

    with open("answer_key_"+ model + ".pkl", "wb") as f:
        pickle.dump(answer_key, f)

    print(f"Output dictionary saved to output_file_{model}.pkl")
    print(f"Answer key saved to answer_key_{model}.pkl")

start_time = time.time()
parse_data()
end_time = time.time()
total_time = end_time - start_time

if test:
    with open("test_results_" + model + ".txt", "w", encoding="utf-8") as file:
        for prompt_id in output_dict:
            model_answer = output_dict.get(prompt_id, "N/A")
            correct_answer = answer_key.get(prompt_id, "N/A")
            file.write(f"{prompt_id} {model_answer} {correct_answer}\n")


mark_invalid_answers(output_dict)
save_output(output_dict, answer_key)
print(f"Total Execution Time for Llama Process: {total_time:.2f} seconds")