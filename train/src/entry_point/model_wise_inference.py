import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------------
# Utility Functions
# ----------------------------------

def load_config(model_name: str, config_path: str) -> dict:
    """Loads model inference config from JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not found in {config_path}")

    return config[model_name]


def clean_response(response: str):
    """Extracts assistant answer from decoded text."""
    res = re.search(r"\n\nAssistant:(.*)$", response, flags=re.DOTALL)
    return res.group(1).strip().lower() if res else "Incomplete response"

def format_input_for_qwen(instruction: dict, query_key = "normal_query", add_strict_prompt = True, auxiliary_prompt = "", remove_answer_string = True) -> str:
    """Formats input prompt with auxiliary prompt and strict response instructions for Qwen models."""

    example1 = instruction["example"].get("example1", {})
    example2 = instruction["example"].get("example2", {})
    prompt = (
        "Example:\n"
        "Input: " + example1.get("input", "") + "\n"
        "Output: " + example1.get("output", "") + "\n\n"
        "Example:\n"
        "Input: " + example2.get("input", "") + "\n"
        "Output: " + example2.get("output", "") + "\n\n"
        "Now classify the following profile:\n\n"
        "Input: " + "'" + instruction.get("text", "")+ "'\n"
        "Final Answer:"
    )
    if remove_answer_string:
        prompt = re.sub(r'Answer:\s*$', '', prompt).strip()
    if auxiliary_prompt:
        prompt += " " + auxiliary_prompt
    if add_strict_prompt:
        choices = [ "'" + item + "'" for item in instruction.get("choices", [])]
        prompt += f" FINAL ANSWER MUST BE EXACTLY ONE WORD FROM {' OR '.join(choices)}:"
    
    return prompt

def run_qwen_inference(model, tokenizer, user_text, generation_config, choices=["yes", "no"]):
    """
    Performs strict yes/no classification using Qwen/Qwen2/Qwen2.5.
    Ensures NO chain-of-thought and NO explanation leaks.
    """
    
    # System message to force deterministic, short, no-CoT output
    choices_str = " or ".join([f"'{choice}'" for choice in choices])
    system_prompt = (
        "You are a strict binary classifier. "
        f"You must respond with EXACTLY one word: {choices_str}. "
        "Do NOT output any chain-of-thought. "
        "Do NOT output <think> tags. "
        "Do NOT output any explanation. "
        f"Only output {choices_str}."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

    # Convert to Qwen chat format
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        truncation=True,
        add_generation_prompt=True
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoded,
            **generation_config
        )

    # Decode
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Clean unwanted garbage (reasoning tags, role tokens, headers)
    for bad in ["assistant", "<think>", "</think>", "## Thinking", "```", "<|im_end|>"]:
        raw = raw.replace(bad, "")

    clean = raw.strip().lower()

    # Keep ONLY "yes"/"good" or "no"/"bad"
    if choices[0] in clean:
        return choices[0]
    if choices[1] in clean:
        return choices[1]

    # Fallback (should never happen)
    return clean

def format_input(instruction: dict, query_key = "normal_query", add_strict_prompt = True, auxiliary_prompt = "", remove_answer_string = True) -> str:
    """Formats input prompt with auxiliary prompt and strict response instructions."""

    prompt = instruction.get(query_key, "")
    if remove_answer_string:
        prompt = re.sub(r'Answer:\s*$', '', prompt).strip()
    if auxiliary_prompt:
        prompt += " " + auxiliary_prompt
    if add_strict_prompt:
        choices = [ "'" + item + "'" for item in instruction.get("choices", [])]
        prompt += f" FINAL ANSWER MUST BE EXACTLY ONE WORD FROM {' OR '.join(choices)}:"
    
    return prompt

def transform_dict(data: dict, query_key="normal_query") -> dict:
    """Transforms inference output into evaluation format."""
    doc_id = data.get("id")
    query = data.get(query_key, "")

    llm_response = data.get("llm_response", "response not found")
    if llm_response == "response not found":
        return {"doc_id": doc_id, "error": "llm_response key not found"}

    predicted_answer = llm_response # clean_response(llm_response)
    truth = data.get("answer", "").strip().lower()

    acc = "1.0" if predicted_answer == truth else "0.0"
    actual_result_set = {x.strip().lower() for x in data.get("choices", [])}

    if not actual_result_set:
        return {"doc_id": doc_id, "error": "missing_choices"}

    missing = "0" if predicted_answer in actual_result_set else "1"
    metric_tuple = (predicted_answer, truth)

    return {
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


def run_model_inference(model, tokenizer, full_prompt, generation_config):
    inputs = tokenizer(
        full_prompt,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_config)
    generation_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generation_output


# ----------------------------------
# Main
# ----------------------------------

if __name__ == "__main__":

    # ---------- File paths ----------
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    split_data_dir = os.path.join(project_dir, "data", "split_data")
    # ---------- Test Data Path ----------
    # German: ("German_credit_scoring", "german_gender_bias.jsonl"), ("German_credit_scoring", "german_age_bias.jsonl"), ("German_credit_scoring", "german_foreign_bias.jsonl")
    # ccFraud: ("ccFraud_fraud_detection", "ccFraud_gender_bias.jsonl")
    # Travel Insurance: ("Travel_Insurance", "travel_insurance_age_bias.jsonl")
    test_data_path = os.path.join(
        split_data_dir,
        "German_credit_scoring",
        "german_foreign_bias.jsonl"
    )

    # ---------------------------
    # Load Model Config
    # ---------------------------
    model_name = "TheFinAI/Fin-o1-8B"
    config_path = os.path.join(current_dir, "model_inference_config.json")

    model_cfg = load_config(model_name, config_path)
    query_key = model_cfg["query_key"]
    generation_config = model_cfg["generation_config"]
    auxiliary_prompt = model_cfg["auxiliary_prompt"]
    remove_answer_string = model_cfg.get("remove_answer_string", True)
    
    # ---------- Output Path ----------
    # German: ("German_credit_scoring", "german_gender_zero_shot.json"), ("German_credit_scoring", "german_age_zero_shot.json"), ("German_credit_scoring", "german_foreign_zero_shot.json")
    # ccFraud: ("ccFraud_fraud_detection", "ccFraud_gender_zero_shot.json")
    # Travel Insurance: ("Travel_Insurance", "travel_insurance_age_zero_shot.json")
    llm_output_path = os.path.join(
        project_dir,
        "inference",
        "model_inference",
        model_name.replace("/", "_"),
        "German_credit_scoring",
        "german_foreign_zero_shot.json"
    )

    # ---------- Load Test Data ----------
    instruction_list = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            instruction_list.append(json.loads(line))

    # ---------- Device ----------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # ---------- Load Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---------- Load Model ----------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,
        low_cpu_mem_usage=False
    )

    model.to(device)
    model.eval()


    # ---------- Single Test Example ----------
    # example_query = "What is the result of 3-5? Just tell the answer."
    # example_query = auxiliary_prompt + example_query
    # inputs = tokenizer(example_query, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     output_ids = model.generate(**inputs, **generation_config)
    # print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


    # ---------- Full Inference ----------
    llm_response = []

    for instruction in instruction_list:

        # ---------- Generate Response ----------
        # full_prompt = format_input(instruction, query_key=query_key, add_strict_prompt=False, auxiliary_prompt=auxiliary_prompt, remove_answer_string=remove_answer_string)
        # inputs = tokenizer(
        #     full_prompt,
        #     truncation=True,
        #     return_tensors="pt"
        # ).to(model.device)
        # with torch.no_grad():
        #     output_ids = model.generate(**inputs, **generation_config)
        # generation_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if model_name == "TheFinAI/Fin-o1-8B":
            full_prompt = format_input_for_qwen(instruction, query_key=query_key, add_strict_prompt=False, auxiliary_prompt=auxiliary_prompt, remove_answer_string=remove_answer_string)
            print("Full Prompt:\n", full_prompt)
            generation_output = run_qwen_inference(model, tokenizer, full_prompt, generation_config, choices=instruction.get("choices", ["yes", "no"]))
        else:
            full_prompt = format_input(instruction, query_key=query_key, add_strict_prompt=False, auxiliary_prompt=auxiliary_prompt, remove_answer_string=remove_answer_string)
            generation_output = run_model_inference(model, tokenizer, full_prompt, generation_config)

        print(generation_output)
        print("True output:", instruction["answer"])
        print("-" * 100)

        temp = transform_dict(
            {**instruction, "llm_response": generation_output, "model_name": model_name},
            query_key=query_key
        )
        llm_response.append(temp)

    # ---------- Save Results ----------
    os.makedirs(os.path.dirname(llm_output_path), exist_ok=True)
    with open(llm_output_path, "w", encoding="utf-8") as f:
        json.dump(llm_response, f, indent=4, ensure_ascii=False)

    print(f"LLM generation results saved to {llm_output_path}")