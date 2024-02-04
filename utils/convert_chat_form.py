import os
import json
import pandas as pd

#SYSTEM_PROMPT = "<<SYS>> Only provide the natural language version of the latex code in your response. Do not include anything redundant. <</SYS>>"

def format_message(inst,response):
    return f"<s>[INST] {inst} [/INST] {response} </s>"

def convert_to_format(dataset_path):
    output=[]
    with open(dataset_path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            inst = json_object.get("latex_entry")
            response = json_object.get("wa_nli")
            output.append(format_message(inst,response))
    return output

def main(dataset_path,output_path):
    data = convert_to_format(dataset_path)
    df = pd.DataFrame({"text": data})
    df.to_parquet(output_path, index=False)