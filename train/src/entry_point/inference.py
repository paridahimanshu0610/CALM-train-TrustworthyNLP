import numpy as np
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json, os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--mode', type=str, choices=['base', 'lora'], default='lora')
parser.add_argument('--model_name', type=str, default='CALM')
parser.add_argument('--query_key', type=str, default='chat_query')
args = parser.parse_args()


max_new_tokens = 4096
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)


# ---------- File paths ----------
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..")) 

# Input test data path
split_data_dir = os.path.join(project_dir, 'data', 'split_data')
test_data_path = os.path.join(split_data_dir, 'Polish_bankruptcy_prediction', "test.jsonl")

# Output LLM generation results path
llm_output_path = os.path.join(project_dir, 'inference', 'models', 'CALM', 'Polish_bankruptcy_prediction', "Polish_bankruptcy_prediction.json")

# ---------- Load JSON ----------
instruction_list = []
with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        instruction_list.append(json.loads(line))
instruction_list = instruction_list[25:43]  # limit entries for testing


# instruction_list = [
#     "Human: \nXiaoming has 12 oranges. He wants to share them equally among his 4 friends. How many oranges does each person get? \n\nAssistant:\n",
#     "Human: \nHere is an elementary school math problem: Xiaoming has 3 pet cats and 2 pet dogs, while Xiaohua has 4 pet cats and 1 pet dog. Who has more pets?\n\nAssistant:\n",
#     "Human: \nProblem: Xiaoming has 5 balls, he gives 2 balls to Xiaohong, how many balls does he have left?\n\nAssistant:\n",
#     "Human: \nWhat is 2+3?\n\nAssistant:\n"
# ]


def clean_response(response:str):
    res = re.search(r"\n\nAssistant:(.*)$", response, flags=re.DOTALL)
    
    if res:
        res = res.group(1).strip().lower()
    else:
        res = 'Incomplete response'
        
    return res


def transform_dict(data: dict, query_key = "chat_query") -> dict:
    """
    Transforms the input dictionary into the required output format.
    """
    doc_id = data.get("id")
    query = data.get(query_key, "")
    
    llm_response = data.get("llm_response", "response not found")
    if llm_response == "response not found":
        return {"doc_id": doc_id, "error": "llm_response key not found"}
    predicted_answer = clean_response(llm_response)
    
    truth = data.get("answer", "").strip().lower()

    # Compute accuracy
    acc = "1.0" if predicted_answer == truth else "0.0"
    
    # Compute missing (1 if predicted answer is not 'good' or 'bad')
    actual_result_set = {x.strip().lower() for x in data.get("choices", [])}
    if len(actual_result_set) == 0:
        return {"doc_id": doc_id, "error": "missing_choices"}
    missing = "0" if predicted_answer in actual_result_set else "1"

    # F1, macro_f1, and MCC (all same tuple format)
    metric_tuple = (predicted_answer, truth)

    transformed = {
        "doc_id": doc_id,
        "prompt_0": query,
        "model_name": data.get("model_name", "unknown"),
        "llm_response": llm_response,
        "logit_0": predicted_answer,
        "truth": truth,
        "acc": acc,
        "missing": missing,
        "f1": str(metric_tuple),
        "macro_f1": str(metric_tuple),
        "mcc": str(metric_tuple)
    }

    return transformed 

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

    # Single entry processing (for testing)
    # for instruction in instruction_list:
    #     inputs = tokenizer(instruction["chat_query"], max_length=max_new_tokens,truncation=True,return_tensors="pt")
    #     generation_output = model.generate(
    #         input_ids = inputs["input_ids"].to(device), 
    #         **generation_config
    #     )[0]

    #     generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
    #     print(generate_text)
    #     print("-"*100)

    # Process in batches for efficiency
    batch_size = 8   # tune this depending on your GPU memory
    llm_response = []
    for i in range(0, len(instruction_list), batch_size):
        batch = instruction_list[i:i+batch_size]
        prompts = [item[args.query_key] for item in batch]

        inputs = tokenizer(prompts, max_length=max_new_tokens, padding=True, truncation=True, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            **generation_config
        )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Process and store each response
        temp = [transform_dict({**batch[i], "llm_response": t, "model_name": args.model_name}, query_key=args.query_key) for i, t in enumerate(texts)]
        llm_response.extend(temp)

        # Printing results for debugging
        # for i, t in enumerate(texts):
        #     print(t)
        #     print("-"*100)
        #     temp = transform_dict({**batch[i], "llm_response": t, "model_name": args.model_name}, query_key=args.query_key)
        #     llm_response.append(temp)
        
    
    # Save LLM generation results to JSON
    os.makedirs(os.path.dirname(llm_output_path), exist_ok=True)
    with open(llm_output_path, "w", encoding="utf-8") as f:
        json.dump(llm_response, f, indent=4, ensure_ascii=False)

    print(f"LLM generation results saved to {llm_output_path}")