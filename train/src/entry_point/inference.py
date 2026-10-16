import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
import argparse
import json, os
import re

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
parser.add_argument('--use_system_message', action='store_true',
                    help="Prepend a system message telling the model to answer with exactly one of the choices")

parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
parser.add_argument('--project_dir', type=str, default=None,
                    help="Folder containing data/split_data. Defaults to 3 levels above this script.")

args = parser.parse_args()
if args.use_system_message and not args.model_name.endswith("-sysmsg"):
    args.model_name += "-sysmsg"

generation_config = dict(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
)

# =============================================================================
# Paths
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# NOTE: adjust this if you move the script to a different depth in the repo.
if args.project_dir:
    project_dir = os.path.abspath(args.project_dir)
else:
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

#=============================================================================
# MPS support to device/dtype selection
#=============================================================================
def pick_device(pref):
    if pref != 'auto':
        return pref
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# =============================================================================
# Helpers (same logic as the local Mac script)
# =============================================================================
def clean_response(response: str, choices=None):
    r = response.strip().lower()
    if choices:
        hits = [c for c in choices if re.search(rf"\b{re.escape(c)}\b", r)]
        if len(hits) == 1:          # exactly one label mentioned -> unambiguous
            return hits[0]
    r = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", r)   # strip quotes/periods
    return r if r else "incomplete response"


def transform_dict(data: dict, query_key: str) -> dict:
    doc_id = data.get("id")
    query = data.get(query_key, "")

    llm_response = data.get("llm_response", "response not found")
    if llm_response == "response not found":
        return {"doc_id": doc_id, "error": "llm_response key not found"}

    # Build the normalized choice list first so clean_response can match against it
    choices = [x.strip().lower() for x in data.get("choices", [])]
    if not choices:
        return {"doc_id": doc_id, "error": "missing_choices"}
    actual_result_set = set(choices)

    predicted_answer = clean_response(llm_response, choices)

    truth = data.get("answer", "").strip().lower()
    acc = "1.0" if predicted_answer == truth else "0.0"
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

def build_system_message(choices):
    """Builds the instruction from the dataset's own choices."""
    quoted = [f"'{c}'" for c in choices]
    if len(quoted) == 1:
        options = quoted[0]
    elif len(quoted) == 2:
        options = f"{quoted[0]} or {quoted[1]}"
    else:
        options = ", ".join(quoted[:-1]) + f", or {quoted[-1]}"
    return f"Answer with exactly one word: {options}. Do not explain."


def build_prompts(chunk, query_key, tokenizer, system_message=None):
    prompts = []
    for item in chunk:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": item[query_key]})
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts

@torch.no_grad()
def run_batch_scoring(batch, query_key, model, tokenizer, choices, batch_size,
                      system_message=None):
    choice_ids = []
    for c in choices:
        variants = {c, c.capitalize()}
        ids = {tokenizer.encode(v, add_special_tokens=False)[0] for v in variants}
        choice_ids.append(sorted(ids))

    all_labels, all_extras = [], []
    for i in range(0, len(batch), batch_size):
        chunk = batch[i:i + batch_size]
        prompts = build_prompts(chunk, query_key, tokenizer, system_message)
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        logits = model(**inputs).logits[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        scores = torch.stack(
            [torch.logsumexp(logprobs[:, ids], dim=-1) for ids in choice_ids], dim=-1
        )
        norm_probs = torch.softmax(scores, dim=-1)
        raw_mass = scores.exp()
        total_mass = raw_mass.sum(dim=-1)

        best = scores.argmax(dim=-1).tolist()
        for j, b in enumerate(best):
            all_labels.append(choices[b])
            all_extras.append({
                "choice_probs_normalized": {c: round(norm_probs[j, k].item(), 6) for k, c in enumerate(choices)},
                "choice_probs_raw": {c: round(raw_mass[j, k].item(), 6) for k, c in enumerate(choices)},
                "choice_mass_total": round(total_mass[j].item(), 6),
            })
    return all_labels, all_extras

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
            prompts, padding=True, truncation=True, return_tensors="pt",
            add_special_tokens=False,   # chat template already added <|begin_of_text|>
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
    device = pick_device(args.device)
    load_type = torch.float32 if device == 'cpu' else torch.float16
    print(f"Using device: {device}, dtype: {load_type}")

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
    # device_map="auto" only on CUDA; on MPS/CPU load normally and move.
    device_map = "auto" if device == 'cuda' else None

    if args.mode == "base":
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=load_type, 
            device_map=device_map,
            config=model_config,
        )
    elif args.mode == "lora":
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required when --mode lora")
        print("Loading LoRA model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, device_map=device_map
        )
        model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)

    if device_map is None:
        model.to(device)

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

        choices = [c.strip().lower() for c in instruction_list[0]["choices"]]
        system_message = build_system_message(choices) if args.use_system_message else None
        if system_message:
            print(f"  System message: {system_message}")

        raw_outputs, extras = run_batch_scoring(
            instruction_list, query_key, model, tokenizer, choices, args.batch_size,
            system_message=system_message,
        )

        results = []
        for item, raw_out, extra in zip(instruction_list, raw_outputs, extras):
            record = transform_dict(
                {**item, "llm_response": raw_out, "model_name": args.model_name},
                query_key=query_key,
            )
            record.update(extra)                        # adds the three new fields
            record["system_message"] = system_message   # None when the flag is off
            results.append(record)
            if args.smoke_test:
                print("Input query:", item[query_key])
                print("Model output:", raw_out, " | True output:", item.get('answer'))
                print("Normalized:", extra["choice_probs_normalized"], "| Total mass:", extra["choice_mass_total"])
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