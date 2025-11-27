#!/bin/bash
# ===============================================
# Mac M4 LoRA Training Script for LLaMA-2 7B Chat
# ===============================================

# Set paths
export ABS_PATH="/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP"
export PYTHONPATH="$ABS_PATH/train"
model_name_or_path="$ABS_PATH/models/Llama-2-7b-chat-hf"

train_file="$ABS_PATH/train/data/CRA-resample-train4w.json"
validation_file="$ABS_PATH/train/data/CRA-resample-dev3k.json"
output_dir="$ABS_PATH/train/saved_models2/llama2_m4_lora"
lora_config_llama="$ABS_PATH/train/configs/lora_config_llama.json"
mkdir -p ${output_dir}

cache_dir="$PYTHONPATH/hf_cache_dir_2"
log_dir="$PYTHONPATH/train_log_dir_2"
mkdir -p ${cache_dir} ${log_dir}

cutoff_len=2048  # Maximum sequence length

# =========================
# Training parameters
# =========================
per_device_train_batch_size=1     # Must be small due to limited GPU memory
per_device_eval_batch_size=1
gradient_accumulation_steps=2     # To increase effective batch size
num_train_epochs=5
learning_rate=3e-4
weight_decay=1e-5
warmup_ratio=0.01
lr_scheduler_type="cosine"
logging_steps=10
# evaluation_strategy="steps"
save_total_limit=3
gradient_checkpointing=True
use_lora=True
bf16=False  # MPS does not support bf16; float16 or float32 only

# =========================
# Run training
# =========================
echo "Starting LoRA training on Mac M4 (MPS)..."

python3 /Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/src/entry_point/sft_train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama True \
    --use_lora ${use_lora} \
    --lora_config ${lora_config_llama} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps ${logging_steps} \
    --seed 1234 \
    --gradient_checkpointing ${gradient_checkpointing} \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --bf16 ${bf16} \
    --device "mps" \
    > ${log_dir}/train_m4.log 2>&1 &

echo "Training started. Logs are being written to ${log_dir}/train_m4.log"