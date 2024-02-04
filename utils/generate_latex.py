from openai import OpenAI
import os
import pickle
import json
import random

MODEL = "gpt-3.5-turbo"
PROMPT = "Generate one example of raw latex (don't render the latex) that might appear in a calculus, linear algebra or other college level stem class homework solution. The latex example should not contain words or ellipses and should be somewhat complex. Only include the latex example in your response."

def load_subjects(subjects_path):
  with open(subjects_path, 'r') as json_file:
      return json.load(json_file)

def pick_random_topic(subjects):
   """
   First picks course randomly with equal probability, then picks topic within
   that course randomly with equal probability, returning the course and topic.
   """
   course = random.choice(list(subjects.keys()))
   topic = random.choice(subjects[course])
   return course, topic


def generate_prompt(course,topic):
  """
  Flips a coin to decide 
  -if the response should include real numbers (most responses
  tend to only include variables).
  -if the response should be "somewhat complex"
  """
  use_numbers = random.choice([True,False])
  sc = random.choice([True,False])
  prompt= f"Generate one example of raw latex (don't render the latex) that might appear in a college level {course} homework solution covering {topic}. The latex example should be one line and should not contain sentences or dots denoting an indefinite sequence. Only include the latex example in your response."
  if use_numbers and sc:
    prompt += ' The response should be somewhat complex and include numbers.'
  elif use_numbers:
    prompt += ' Include numbers.'
  elif sc:
    prompt += ' The response should be somewhat complex.'
  return prompt

def generate_responses(prompt,client):
  message=[{"role": "user", "content": prompt}]
  response = client.chat.completions.create(
    model=MODEL,
    messages = message,
    temperature=0.5)
  return response

def save_as_pickle(result,output_path):
  with open(output_path, 'wb') as file:
    pickle.dump(result, file)

def main(basedir,output_path,num,api_key):
  subjects_path= os.path.join(basedir,'misc','subjects.json')
  subjects = load_subjects(subjects_path)
  client = OpenAI(api_key=api_key)
  result = set()
  for i in range(num):
    course, topic = pick_random_topic(subjects)
    prompt=generate_prompt(course,topic)
    response = generate_responses(prompt, client)
    generated_text = {response.choices[0].message.content}
    if not isinstance(generated_text, set):
       generated_text = {generated_text}
    result = result.union(generated_text)
    if i%50==0:
      print(i)
      save_as_pickle(result,output_path)
  save_as_pickle(result)