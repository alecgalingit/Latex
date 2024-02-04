import utils.generate_latex as generate_latex
import utils.generate_dataset as generate_dataset
import utils.convert_chat_form as convert_chat_form
import os
import random
import json

# Change this!
BASEDIR = '/Users/alecsmac/coding/latex/'
API_KEY = ""
DATASET_SIZE = 820
SPLIT_RATIO  = 0.988

def split_data(dataset_path,train_output_path,valid_output_path,split_ratio):
  with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

  random.shuffle(data)
  split_index = int(len(data) * split_ratio)
  train_data = data[:split_index]
  valid_data = data[split_index:]

  with open(train_output_path, 'w') as train_f:
    for item in train_data:
      train_f.write(json.dumps(item) + '\n')

  with open(valid_output_path, 'w') as valid_f:
    for item in valid_data:
      valid_f.write(json.dumps(item) + '\n')

def create_dataset(basedir=BASEDIR,api_key=API_KEY,size=DATASET_SIZE,split_ratio=SPLIT_RATIO):
  latex_path = os.path.join(BASEDIR,'generated_data','latex.pkl')
  dataset_path = os.path.join(BASEDIR,'generated_data','latex_dataset.jsonl')
  train_path = os.path.join(BASEDIR,'generated_data','train.jsonl')
  validation_path = os.path.join(BASEDIR,'generated_data','validation.jsonl')
  chat_train_path = os.path.join(BASEDIR,'generated_data','chat_train.parquet')

  # Generate latex
  generate_latex.main(BASEDIR,latex_path,size,API_KEY)

  # Generate dataset
  generate_dataset.main(BASEDIR,latex_path,dataset_path,API_KEY)

  # Split into train and validation datasets
  split_data(dataset_path,train_path,validation_path,split_ratio)

  # Convert train data to chat form suitable for Llama
  convert_chat_form.main(train_path,chat_train_path)


def main():
  create_dataset()

if __name__ == "__main__":
  main()
