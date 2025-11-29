#!/bin/bash
#SBATCH --job-name=trustworthy_nlp_calm_inference
#SBATCH --output=/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/inference/inference.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 04:00:00

###################################
# Load the same modules as training
###################################
module purge
module load GCCcore/13.3.0 Python/3.12.3
module load CUDA/12.6.0
module load NCCL/2.22.3-CUDA-12.6.0

###################################
# Activate environment
###################################
source /scratch/user/paridahimanshu0610/trustworthy_nlp/calm_env/bin/activate

###################################
# Project paths
###################################
export ABS_PATH="/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP"
export PYTHONPATH="$ABS_PATH/train"

# Base model (full model)
model_name_or_path="$ABS_PATH/models/Llama-2-7b-chat-hf"

# LoRA checkpoint path
ckpt_path="$ABS_PATH/train/saved_models/CRA-llama2-7b-chat_CRA_0.045M/checkpoint-2804"

# Output directory for inference logs
inference_dir="$ABS_PATH/inference"
mkdir -p ${inference_dir}

echo "Running inference..."
echo "Base Model: ${model_name_or_path}"
echo "LoRA Checkpoint: ${ckpt_path}"

###################################
# Run inference
###################################
python /scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/train/src/entry_point/inference.py \
    --model_name_or_path /scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/models/Llama-2-7b-chat-hf \
    --ckpt_path /scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/train/saved_models/CRA-llama2-7b-chat_CRA_0.045M/checkpoint-2804 \
    --use_lora \
    --mode base \
    --llama \
    > ${inference_dir}/inference_base.log 2>&1

echo "Inference completed!"
