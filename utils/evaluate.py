import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# This is meant to be run in a collab notebook with 

logging.set_verbosity(logging.CRITICAL)


def generate_response(input,system_prompt,model,tokenizer):
  pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
  result = pipe(f"<s>[INST] {system_prompt} {input} [/INST]")
  res = (result[0]['generated_text'])
  marker = "<</SYS>> "
  index_of_marker = res.find(marker)
  return res[index_of_marker + len("marker"):]
   
def fine_tuned_latex_convert(latex_input,model,tokenizer):
  system_prompt = "<<SYS>> Only provide the natural language version of the latex code in your response. Do not include anything redundant. <</SYS>>"
  return generate_response(latex_input,system_prompt,model,tokenizer)

def evaluation_dataset(valid_path,eval_path,model,tokenizer):
  with open(valid_path, 'r') as infile, open(eval_path,'w') as outfile:
    for line in infile:
        data = json.loads(line)
        fine_tuned = fine_tuned_latex_convert(data["latex_entry"],model,tokenizer)
        data["fine_tuned"] = fine_tuned
        updated_line = json.dumps(data)
        outfile.write(updated_line + '\n')


## Get accuracy score with evaluation dataset
        
def generate_accuracy_prompt(gpt,fine_tuned):
  return f"Are the two following expressions in quotes semantically identical? Respond with 1 if yes and 0 if no. First expression: '{gpt}'. Second expression: '{fine_tuned}'."

def get_accuracy_score(gpt,fine_tuned,model,tokenizer):
   system_prompt = "<<SYS>> Your response should only be 1 or 0. Do not include anything else. <</SYS>>"
   input = generate_accuracy_prompt(gpt,fine_tuned)
   return generate_response(input,system_prompt,model,tokenizer)

def evaluation_score(eval_path,model,tokenizer):
  num_accurate,total = 0
  with open(eval_path, 'r') as infile:
    for line in infile:
        data = json.loads(line)
        gpt = data["wa_nli"]
        fine_tuned= data["fine_tuned"]
        accuracy = get_accuracy_score(gpt,fine_tuned,model,tokenizer)
        num_accurate += (accuracy==1)
        total +=1
  return (num_accurate/total)