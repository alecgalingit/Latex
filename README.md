# Latex

The purpose of this repository is to fine-tune the open source language model LLaMa 2 in order to efficientiely convert Latex code to natural language input. The repository contains the following modules:
  * generate_chat_dataset: utilizes the modules in the utils folder to generate a dataset of Latex code and its corresponding natural language input with GPT.
  * train: finetunes LLama 2 with QLoRa. Due to the paramater-efficient fine-tuning (PEFT) techniques applied, this code can be run on a T4 GPU.

A detailed explanation of the project, methodology, and results can be found here: https://docs.google.com/document/d/1DvZoTtFxlb0OgJZhCbLHrwAeSwouk81ut2tfAw7ZMVE/edit?usp=sharing
