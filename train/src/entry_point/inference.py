import numpy as np
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json, os
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--mode', type=str, choices=['base', 'lora'], default='lora')
args = parser.parse_args()


max_new_tokens = 1024
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)

instruction_list = [
    "Human: \nXiaoming has 12 oranges. He wants to share them equally among his 4 friends. How many oranges does each person get? \n\nAssistant:\n",
    "Human: \nHere is an elementary school math problem: Xiaoming has 3 pet cats and 2 pet dogs, while Xiaohua has 4 pet cats and 1 pet dog. Who has more pets?\n\nAssistant:\n",
    "Human: \nProblem: Xiaoming has 5 balls, he gives 2 balls to Xiaohong, how many balls does he have left?\n\nAssistant:\n",
    "Human: \nWhat is 2+3?\n\nAssistant:\n"
]


if __name__ == '__main__':
    load_type = torch.float16 #Sometimes may need torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Uncomment this block if use_lora is needed
    # if args.use_lora:
    #     base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type)
    #     model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=load_type, config=model_config)

    # Use the parameter "mode" to decide whether ot use LoRA or base model
    if args.mode == "base":
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type, config=model_config)
    elif args.mode == "lora":
        print("Loading LoRA model...")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type)
        model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)

    if device==torch.device('cpu'):
        model.float()

    model.to(device)
    model.eval()
    print("Load model successfully")

    for instruction in instruction_list:
        inputs = tokenizer(instruction, max_length=max_new_tokens,truncation=True,return_tensors="pt")
        generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device), 
            **generation_config
        )[0]

        generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
        print(generate_text)
        print("-"*100)
