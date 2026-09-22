import numpy as np
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json, os
import re

max_new_tokens = 5
generation_config = dict(
    do_sample=False,
    max_new_tokens=5
)

# ---------- File paths ----------
current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..")) 

split_data_dir = os.path.join(project_dir, 'data', 'split_data')
test_data_path = os.path.join(split_data_dir, 'ccFraud_fraud_detection', "ccFraud_gender_bias.jsonl") 

llm_output_path = os.path.join(project_dir, 'inference', 'model_inference', 'Llama_baseline', 'ccFraud_fraud_detection', "ccFraud_gender.json")

debug = True

# ---------- Load JSON ----------
instruction_list = []
with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        instruction_list.append(json.loads(line))

# ---------- Set args here for testing ----------
args = {
    # resolved from ~/.cache/huggingface/hub automatically via repo id
    'model_name_or_path': "meta-llama/Llama-3.1-8B-Instruct",
    # TODO: fill in once your Llama-3.1 LoRA fine-tune finishes and you know
    # the checkpoint folder / step number, e.g.:
    # 'CRA-llama3.1-8b-instruct_CRA_debiased/checkpoint-XXXX'
    'ckpt_path': None, # os.path.join(project_dir, 'train', 'saved_models', 'CRA-llama3.1-8b-instruct_CRA_debiased', 'checkpoint-XXXX'),
    'llama': True,
    'mode': 'base',
    'model_name': "Llama-3.1-8B-Instruct-base", # 'CALM',
    # use the untemplated field — apply_chat_template builds the prompt now
    'query_key': 'normal_query'
}


def clean_response(response: str):
    response = response.strip().lower()
    return response if response else 'Incomplete response'


def transform_dict(data: dict, query_key="normal_query") -> dict:
    doc_id = data.get("id")
    query = data.get(query_key, "")

    llm_response = data.get("llm_response", "response not found")
    if llm_response == "response not found":
        return {"doc_id": doc_id, "error": "llm_response key not found"}
    predicted_answer = clean_response(llm_response)

    truth = data.get("answer", "").strip().lower()

    acc = "1.0" if predicted_answer == truth else "0.0"

    actual_result_set = {x.strip().lower() for x in data.get("choices", [])}
    if len(actual_result_set) == 0:
        return {"doc_id": doc_id, "error": "missing_choices"}
    missing = "0" if predicted_answer in actual_result_set else "1"

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
    query,
    model,
    generation_config,
    tokenizer,
    two_interactions=False,
    second_query=None,
):
    """
    Runs inference for one or two Human-Assistant turns, using the model's
    own chat template rather than a hand-rolled "Human:/Assistant:" string.

    Parameters
    ----------
    query : str
        Raw user instruction text (no template applied).
    model : PreTrainedModel
    generation_config : dict
    tokenizer : PreTrainedTokenizer
    two_interactions : bool, optional
    second_query : str, optional
        Raw follow-up user instruction if `two_interactions=True`.

    Returns
    -------
    str
        The assistant's response from the last turn.
    """

    def _generate(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, truncation=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids=inputs["input_ids"], **generation_config)[0]
        # decode only the newly generated tokens, not the echoed-back prompt
        new_tokens = output_ids[inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    messages = [{"role": "user", "content": query}]
    first_output = _generate(messages)

    if not two_interactions:
        return first_output

    if two_interactions and second_query is None:
        raise ValueError(
            "two_interactions=True requires second_query to be provided."
        )

    messages.append({"role": "assistant", "content": first_output})
    messages.append({"role": "user", "content": second_query})
    second_output = _generate(messages)

    return second_output


if __name__ == '__main__':
    load_type = torch.float16

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if device.type in ["cpu", "mps"]:
        load_type = torch.float32  # float16 can lead to incorrect outputs on MPS/CPU

    # Load tokenizer
    if args['llama']:
        tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>" \
            if "<|finetune_right_pad_id|>" in tokenizer.get_vocab() \
            else tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(args['model_name_or_path'])

    # Load model (base or LoRA)
    if args['mode'] == "base":
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=load_type,
            device_map="auto",
            config=model_config
        )
    elif args['mode'] == "lora":
        print("Loading LoRA model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=load_type,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model,
            args['ckpt_path'],
            device_map="auto",
            torch_dtype=load_type
        )

    model.eval()
    print("Loaded model successfully")

    llm_response = []
    for instruction in instruction_list[0:5]:
        generate_text = run_inference(
            query=instruction[args['query_key']],
            model=model,
            generation_config=generation_config,
            tokenizer=tokenizer,
            two_interactions=False
        )
        if debug == True:
            print("Input query:", instruction[args['query_key']])
            print("Model output:", generate_text, " | True output:", instruction['answer'])
            print("-" * 100)
        temp = transform_dict({**instruction, "llm_response": generate_text, "model_name": args['model_name']}, query_key=args['query_key'])
        llm_response.append(temp)

    os.makedirs(os.path.dirname(llm_output_path), exist_ok=True)
    with open(llm_output_path, "w", encoding="utf-8") as f:
        json.dump(llm_response, f, indent=4, ensure_ascii=False)

    print(f"LLM generation results saved to {llm_output_path}")