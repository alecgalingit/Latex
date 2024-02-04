from openai import OpenAI
import os
import pickle
import json

MODEL = "gpt-4"

def load_latex(latex_path):
  with open(latex_path, 'rb') as file:
    return pickle.load(file)


def create_system_prompt():
  """
  Removed addon instructions due to cost constraint.
  """
  # with open(ADDON_INSTRUCT_PATH, 'r') as file:
  #   addon_instructions = file.read()
  example_1 = [r'\int_{0}^{8} \int_{0}^{12} x^2 y \, dx \, dy','integrate x^2 y from x=0 to 12 and y=0 to 8']
  example_2 = [r'\frac{d^2}{dx^2}(y^2 + 2xy)','second derivative of y^2 + 2xy with respect to x']
  system_prompt = f"When latex code is provided, convert it into a natural language input accepted by WolframAlpha and only include that natural language input in your response. For example, input: '{example_1[0]}' has the corresponding natural language input: {example_1[1]}. Similary, the input: '{example_2[0]}' has corresponding natural language input {example_2[1]}. If a formula for a well-known theorem is provided, express that formula fully and do not state it's name."
  return system_prompt

def generate_wa_input(latex_entry,instructions,client):
  message=[{"role": "assistant", "content": instructions},{"role": "user", "content": latex_entry}]
  response = client.chat.completions.create(
    model=MODEL,
    messages = message)
  return response.choices[0].message.content

def write_jsonl(dataset,output_path):
  with open(output_path, "w") as jsonl_file:
    for entry in dataset:
      json_line = json.dumps(entry)
      jsonl_file.write(json_line + "\n")

def main(basedir,latex_path,output_path,api_key):
  client = OpenAI(api_key=api_key)
  #addon_instrict_path = os.path.join(basedir,'misc','addon_instruct.txt')
  latex = load_latex(latex_path)
  system_prompt = create_system_prompt()
  dataset = []
  i=0
  for latex_entry in latex:
    wa_nli = generate_wa_input(latex_entry,system_prompt,client)
    dataset.append({'latex_entry':latex_entry,'wa_nli':wa_nli})
    if i%2==0:
      write_jsonl(dataset,output_path)
    i+=1
  write_jsonl(dataset,output_path)




