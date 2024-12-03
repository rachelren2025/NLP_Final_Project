import pandas as pd
import subprocess
import time
import json

output_dict = {}
answer_key = {}
model = "llama3.2-vision"
# set to True to pass the first z into the model
test = False
z = 100

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

        Instruction: You will receive a message representing the context or facts of a legal case, followed by five 
        possible legal holdings. Your task is to select the holding that is most relevant and aligns with the legal 
        principles or facts in the message. Output only one number and nothing else at all.

        Format:

        Message: <Message> 0: <Option 0> 1: <Option 1> 2: <Option 2> 3: <Option 3> 4: <Option 4> Response: Return only the 
        number (0, 1, 2, 3, or 4) corresponding to the holding that is the most contextually relevant to the message.

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
        if test == True and k == z:
            break

        k += 1

        inp_str = 'Message: <' + i['0'] + '> 0: <' + i['1'] + '> 1: <' + i['2'] + '> 2: <' + i['3'] + '> 3: <' + i['4'] + '> 4: <' + i['5'] + '>'

        print(inp_str)

        prompt_id = i['Unnamed: 0']
        out = prompt_file(inp_str)
        output_dict[prompt_id] = out.strip()
        answer_key[prompt_id] = i['11']

        print("\n\n")

def save_output_json(output_dict):
    if test:
        with open("results/test_output_file_" + model + ".json", "w", encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)
    else:
        with open("results/output_file_" + model + ".json", "w", encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)

    print(f"Output dictionary saved to results/output_file_{model}.json")


start_time = time.time()
parse_data()
end_time = time.time()
total_time = end_time - start_time

save_output_json(output_dict)

print(f"Total Execution Time for Llama Process: {total_time:.2f} seconds")