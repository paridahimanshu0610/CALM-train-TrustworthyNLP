import numpy as np
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json, os
import re

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name_or_path', type=str, required=True)
# parser.add_argument('--ckpt_path', type=str, required=True)
# parser.add_argument('--use_lora', action="store_true")
# parser.add_argument('--llama', action="store_true")
# parser.add_argument('--mode', type=str, choices=['base', 'lora'], default='lora')
# parser.add_argument('--model_name', type=str, default='CALM')
# parser.add_argument('--query_key', type=str, default='chat_query')
# args = parser.parse_args()


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
test_data_path = os.path.join(split_data_dir, 'ccFraud_fraud_detection', "ccFraud_gender_bias.jsonl")

# Output LLM generation results path
llm_output_path = os.path.join(project_dir, 'inference', 'model_inference', 'CALM', 'ccFraud_fraud_detection', "ccFraud_zero_shot.json")

# ---------- Load JSON ----------
instruction_list = []
with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        instruction_list.append(json.loads(line))
# instruction_list = instruction_list[31:35]  # limit entries for testing

# ---------- Set args here for testing ----------
args = {
    'model_name_or_path': os.path.join(project_dir, 'models', 'Llama-2-7b-chat-hf'),  # replace with actual model path,
    'ckpt_path': os.path.join(project_dir, 'train', 'saved_models', 'CRA-llama2-7b-chat_CRA_0.045M', 'checkpoint-7010'),  # replace with actual checkpoint path,
    'llama': True,
    'mode': 'lora',
    'model_name': 'CALM',
    'query_key': 'chat_query'
}


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

def run_inference(
    input_prompt,
    model,
    generation_config,
    tokenizer,
    two_interactions=False,
    second_query=None,
):
    """
    Runs inference for one or two Human–Assistant interactions.

    Parameters
    ----------
    input_prompt : str
        The initial prompt starting with "Human:" and ending with "Assistant:".
    model : PreTrainedModel
        Loaded HF model.
    generation_config : dict
        Generation parameters.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    two_interactions : bool, optional
        Whether to perform a second interaction (default: False).
    second_query : str, optional
        User query for the second interaction if `two_interactions=True`.

    Returns
    -------
    str
        The assistant's response from the last interaction.
    """

    def _generate(prompt):
        """Internal helper for running one inference."""
        inputs = tokenizer(prompt, truncation=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids=inputs["input_ids"], **generation_config)[0]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    # ---- First Interaction ----
    first_output = _generate(input_prompt)

    if not two_interactions:
        # Return the assistant part extracted
        return first_output

    # ---- If two_interactions=True ----
    if two_interactions and second_query is None:
        raise ValueError(
            "two_interactions=True requires second_query to be provided."
        )

    # Append the first assistant output and start second Human query
    second_prompt = (
        f"{first_output}\nHuman: {second_query}\nAssistant:"
    )

    second_output = _generate(second_prompt)

    return second_output

if __name__ == '__main__':
    load_type = torch.float16  # Sometimes may need torch.float32
    
    # >>> CHANGED: Add Apple Silicon MPS support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # >>> CHANGED: If on CPU or MPS, float16 can cause problems → switch to float32
    if device.type in ["cpu", "mps"]:
        load_type = torch.float16 # float16 can lead to incorrect outputs on MPS/CPU

    # Load tokenizer
    if args['llama']:
        tokenizer = LlamaTokenizer.from_pretrained(args['model_name_or_path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args['model_name_or_path'])


    # Load model (base or LoRA)
    if args['mode'] == "base":
        print("Loading base model...")
        # >>> CHANGED: Use load_type that adapts to MPS/CPU safely
        # model = AutoModelForCausalLM.from_pretrained(
        #     args['model_name_or_path'], 
        #     torch_dtype=load_type,
        #     config=model_config
        # )
        model = AutoModelForCausalLM.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=load_type,
            device_map="auto",
            config=model_config 
        )
    elif args['mode'] == "lora":
        print("Loading LoRA model...")
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     args['model_name_or_path'],
        #     torch_dtype=load_type
        # )
        # model = PeftModel.from_pretrained(
        #     base_model, 
        #     args['ckpt_path'],
        #     torch_dtype=load_type
        # )
        # Load 4-bit model
        base_model = AutoModelForCausalLM.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=load_type,
            device_map="auto"
        )
        # Load LoRA adapter on top of the 4-bit model
        model = PeftModel.from_pretrained(
            base_model,
            args['ckpt_path'],
            device_map="auto",
            torch_dtype=load_type
        )   

    ############### Should not be used with device_map="auto" #############
    # >>> CHANGED: MPS cannot reliably handle float16 → convert to float32
    # if device.type == "mps":
    #     model = model.to(torch.float32)
    # if device.type == "cpu":
    #     model = model.float()
    # model.to(device)

    model.eval()
    print("Loaded model successfully")

    # >>> CHANGED: Ensure tokenizer tensors are moved to device properly
    llm_response = []
    for instruction in instruction_list:
        # inputs = tokenizer(
        #     instruction["chat_query"],
        #     max_length=max_new_tokens,
        #     truncation=True,
        #     return_tensors="pt"
        # ).to(model.device)   

        # generation_output = model.generate(
        #     input_ids=inputs["input_ids"],
        #     **generation_config
        # )[0]

        # generate_text = tokenizer.decode(
        #     generation_output,
        #     skip_special_tokens=True
        # )

        generate_text = run_inference(
            input_prompt=instruction["chat_query"],
            model=model,
            generation_config=generation_config,
            tokenizer=tokenizer,
            two_interactions=False
        )
        print(generate_text)
        print("True output:", instruction['answer'])
        print("-" * 100)
        temp = transform_dict({**instruction, "llm_response": generate_text, "model_name": args['model_name']}, query_key=args['query_key'])
        llm_response.append(temp)


    # Process in batches for efficiency
    # batch_size = 2   # tune this depending on your GPU memory
    # llm_response = []
    # for i in range(0, len(instruction_list), batch_size):
    #     batch = instruction_list[i:i+batch_size]
    #     prompts = [item[args['query_key']] for item in batch]

    #     inputs = tokenizer(prompts, max_length=max_new_tokens, padding=True, truncation=True, return_tensors="pt").to(model.device)

    #     outputs = model.generate(
    #         **inputs,
    #         **generation_config
    #     )

    #     texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #     # Process and store each response
    #     # temp = [transform_dict({**batch[i], "llm_response": t, "model_name": args['model_name']}, query_key=args['query_key']) for i, t in enumerate(texts)]
    #     # llm_response.extend(temp)

    #     # Printing results for debugging
    #     for i, t in enumerate(texts):
    #         print(t)
    #         print("-"*100)
    #         temp = transform_dict({**batch[i], "llm_response": t, "model_name": args['model_name']}, query_key=args['query_key'])
    #         llm_response.append(temp)
        
    
    # Save LLM generation results to JSON
    os.makedirs(os.path.dirname(llm_output_path), exist_ok=True)
    with open(llm_output_path, "w", encoding="utf-8") as f:
        json.dump(llm_response, f, indent=4, ensure_ascii=False)

    print(f"LLM generation results saved to {llm_output_path}")
