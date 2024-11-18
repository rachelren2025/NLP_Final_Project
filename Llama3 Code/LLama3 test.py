import pandas as pd
import subprocess
import json

output_dict = {}
answer_key = {}


def prompt_file(inp):
    # Send input to the process and get the output

    # Start the process
    proc = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,  # Enables sending input
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard errors
        text=True  # Inputs and outputs are handled as text
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
        # Only pass the first 5 into llama
        if k > 5:
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


parse_data()
print(output_dict)
print(answer_key)
