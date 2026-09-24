import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
import argparse
import json, os

# =============================================================================
# CLI args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True,
                     help="Local path (or HF repo id) to the base model, e.g. Llama-3.1-8B-Instruct")
parser.add_argument('--ckpt_path', type=str, default=None,
                     help="LoRA checkpoint path. Required if --mode lora")
parser.add_argument('--mode', type=str, choices=['base', 'lora'], default='base')
parser.add_argument('--model_name', type=str, default='Llama-3.1-8B-Instruct-base',
                     help="Tag written into the output json + used as an output subfolder name")

# Which files to run
parser.add_argument('--file_types', type=str, choices=['test', 'bias', 'both'], default='both',
                     help="Run inference on test.jsonl files, *_bias.jsonl files, or both")
parser.add_argument('--query_key_test', type=str, default='chat_query',
                     help="Field name to use as the prompt for test.jsonl files")
parser.add_argument('--query_key_bias', type=str, default='normal_query',
                     help="Field name to use as the prompt for *_bias.jsonl files")

# Smoke test
parser.add_argument('--smoke_test', action='store_true',
                     help="If set, only run on --smoke_test_n records per file, and tag output filenames")
parser.add_argument('--smoke_test_n', type=int, default=5)

# Generation / runtime
parser.add_argument('--max_new_tokens', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--llama', action='store_true', default=True)

args = parser.parse_args()

generation_config = dict(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
)

# =============================================================================
# Paths
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# NOTE: adjust this if you move the script to a different depth in the repo.
project_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

split_data_dir = os.path.join(project_dir, 'data', 'split_data')
inference_root = os.path.join(project_dir, 'inference', 'model_inference', args.model_name)


# =============================================================================
# File discovery
# =============================================================================
def discover_files(split_data_dir, file_types):
    """
    Walks data/split_data/<dataset>/ and returns a list of
    (dataset_name, filename, full_path, kind) tuples, where kind is
    'test' or 'bias'.
    """
    found = []
    for dataset_name in sorted(os.listdir(split_data_dir)):
        dataset_path = os.path.join(split_data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        for fname in sorted(os.listdir(dataset_path)):
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(dataset_path, fname)
            if fname == 'test.jsonl' and file_types in ('test', 'both'):
                found.append((dataset_name, fname, fpath, 'test'))
            elif fname.endswith('_bias.jsonl') and file_types in ('bias', 'both'):
                found.append((dataset_name, fname, fpath, 'bias'))
    return found


# =============================================================================
# Helpers (same logic as the local Mac script)
# =============================================================================
def clean_response(response: str):
    response = response.strip().lower()
    return response if response else 'Incomplete response'


def transform_dict(data: dict, query_key: str) -> dict:
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
        "mcc": str(metric_tuple),
    }


def run_batch_inference(batch, query_key, model, tokenizer, generation_config, batch_size):
    """
    Runs chat-template batched inference over `batch` (a list of dicts),
    batch_size at a time. Returns a list of raw decoded response strings,
    same order as `batch`.
    """
    all_outputs = []
    for i in range(0, len(batch), batch_size):
        chunk = batch[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item[query_key]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in chunk
        ]
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)

        output_ids = model.generate(**inputs, **generation_config)

        # left-padding means every sequence's prompt ends at the same index
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[:, prompt_len:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_outputs.extend([t.strip() for t in texts])

    return all_outputs


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        load_type = torch.float32
        print("Using CPU")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.llama:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            "<|finetune_right_pad_id|>"
            if "<|finetune_right_pad_id|>" in tokenizer.get_vocab()
            else tokenizer.eos_token
        )
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    # ---- Model ----
    if args.mode == "base":
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=load_type,
            device_map="auto",
            config=model_config,
        )
    elif args.mode == "lora":
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required when --mode lora")
        print("Loading LoRA model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model, args.ckpt_path, device_map="auto", torch_dtype=load_type
        )

    model.eval()
    print("Loaded model successfully")

    # ---- Discover files ----
    files_to_run = discover_files(split_data_dir, args.file_types)
    if not files_to_run:
        raise SystemExit(f"No matching files found under {split_data_dir} for file_types={args.file_types}")

    print(f"Found {len(files_to_run)} file(s) to run inference on:")
    for dataset_name, fname, fpath, kind in files_to_run:
        print(f"  [{kind}] {fpath}")

    # ---- Run per file ----
    for dataset_name, fname, fpath, kind in files_to_run:
        query_key = args.query_key_test if kind == 'test' else args.query_key_bias

        instruction_list = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instruction_list.append(json.loads(line))

        if args.smoke_test:
            instruction_list = instruction_list[:args.smoke_test_n]

        print(f"\n=== Running [{kind}] {dataset_name}/{fname} "
              f"({len(instruction_list)} records{' - SMOKE TEST' if args.smoke_test else ''}) ===")

        if not instruction_list:
            print(f"  WARNING: {fname} is empty, skipping.")
            continue
        if query_key not in instruction_list[0]:
            print(f"  WARNING: query_key '{query_key}' not found in first record of {fname}, skipping.")
            continue

        raw_outputs = run_batch_inference(
            instruction_list, query_key, model, tokenizer, generation_config, args.batch_size
        )

        results = []
        for item, raw_out in zip(instruction_list, raw_outputs):
            record = transform_dict(
                {**item, "llm_response": raw_out, "model_name": args.model_name},
                query_key=query_key,
            )
            results.append(record)
            if args.smoke_test:
                print("Input query:", item[query_key])
                print("Model output:", raw_out, " | True output:", item.get('answer'))
                print("-" * 100)

        # ---- Output path ----
        out_dir = os.path.join(inference_root, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(fname)[0]
        if args.smoke_test:
            stem += "_smoketest"
        out_path = os.path.join(out_dir, f"{stem}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Saved results to {out_path}")

    print("\nAll requested inference runs completed.")