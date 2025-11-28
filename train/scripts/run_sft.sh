#!/bin/bash
#SBATCH --job-name=trustworthy_nlp_calm
#SBATCH --output=/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/train/trustworthy_nlp_calm.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 56:00:00

###############################
# Load the same modules used to create the venv
###############################
module purge
module load GCCcore/13.3.0 Python/3.12.3
module load CUDA/12.6.0
module load NCCL/2.22.3-CUDA-12.6.0

###############################
# Activate environment
###############################
source /scratch/user/paridahimanshu0610/trustworthy_nlp/calm_env/bin/activate

###############################
# Grace HPRC – GPU settings
###############################
# DO NOT set CUDA_VISIBLE_DEVICES manually — Slurm handles this.

export WANDB_PROJECT=CRA-llama2-7b-chat
export WANDB_RUN_ID=CRA_0.045M
# export WANDB_RESUME=allow
# export WANDB_API_KEY=YOUR_WANDB_KEY   # <-- Replace manually

###############################
# Project paths on GRACE
###############################
export ABS_PATH="/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP"
export PYTHONPATH="$ABS_PATH/train"

model_name_or_path="$ABS_PATH/models/Llama-2-7b-chat-hf"

train_file="$ABS_PATH/train/data/CRA-resample-train4w.json"
validation_file="$ABS_PATH/train/data/CRA-resample-dev3k.json"

output_dir="$ABS_PATH/train/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir="$ABS_PATH/train/hf_cache_dir"
log_dir="$ABS_PATH/train/train_log_dir"
mkdir -p ${cache_dir}
mkdir -p ${log_dir}

lora_config_llama="$ABS_PATH/train/configs/lora_config_llama.json"
deepspeed_config_stage3="$ABS_PATH/train/configs/deepspeed_config_stage3.json"

cutoff_len=2048

echo "Logging to: ${log_dir}/train.log"

###############################
# Training (2 × A100 GPUs)
###############################
torchrun --nproc_per_node=2 /scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/train/src/entry_point/sft_train.py \
    --model_name_or_path ${model_name_or_path} \
    --bf16 True \
    --llama True \
    --use_lora True \
    --deepspeed ${deepspeed_config_stage3} \
    --lora_config ${lora_config_llama} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    > ${log_dir}/train.log 2>&1