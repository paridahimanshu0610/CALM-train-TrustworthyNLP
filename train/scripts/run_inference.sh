#!/bin/bash
#SBATCH --job-name=trustworthy_nlp_llama31_inference
#SBATCH --output=/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/inference/inference_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 04:00:00

###################################
# Usage:
#   sbatch run_inference.sh <FILE_TYPES> <SMOKE_TEST> <SMOKE_TEST_N>
#
#   FILE_TYPES    : test | bias | both   (default: both)
#   SMOKE_TEST    : true | false         (default: false)
#   SMOKE_TEST_N  : integer              (default: 5, only used if SMOKE_TEST=true)
#
# Examples:
#   sbatch run_inference.sh                    # full run, both test + bias files
#   sbatch run_inference.sh test true 5        # smoke test, test.jsonl files only, 5 records each
#   sbatch run_inference.sh bias false         # full run, *_bias.jsonl files only
###################################
FILE_TYPES="${1:-both}"
SMOKE_TEST="${2:-false}"
SMOKE_TEST_N="${3:-5}"

###################################
# Load modules (same as training)
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

# Base model - must already be downloaded locally (see note below).
model_name_or_path="$ABS_PATH/models/Llama-3.1-8B-Instruct"

inference_dir="$ABS_PATH/inference"
mkdir -p "${inference_dir}"

echo "Running inference..."
echo "Base Model: ${model_name_or_path}"
echo "File types: ${FILE_TYPES}"
echo "Smoke test: ${SMOKE_TEST} (n=${SMOKE_TEST_N})"

###################################
# Build optional smoke-test flags
###################################
SMOKE_FLAGS=""
if [ "${SMOKE_TEST}" = "true" ]; then
    SMOKE_FLAGS="--smoke_test --smoke_test_n ${SMOKE_TEST_N}"
fi

###################################
# Run inference
###################################
python "${ABS_PATH}/train/src/entry_point/inference.py" \
    --model_name_or_path "${model_name_or_path}" \
    --mode base \
    --llama \
    --model_name Llama-3.1-8B-Instruct-base \
    --file_types "${FILE_TYPES}" \
    --query_key_test chat_query \
    --query_key_bias normal_query \
    --batch_size 8 \
    ${SMOKE_FLAGS} \
    > "${inference_dir}/inference_${FILE_TYPES}_${SMOKE_TEST}.log" 2>&1

echo "Inference completed!"

###################################
# NOTE ON DOWNLOADING Llama-3.1-8B-Instruct
###################################
# meta-llama/Llama-3.1-8B-Instruct is a gated model. Grace compute nodes
# typically have no/limited outbound internet, so download it ahead of time
# on the LOGIN node (not inside this job):
#
#   module load GCCcore/13.3.0 Python/3.12.3
#   source /scratch/user/paridahimanshu0610/trustworthy_nlp/calm_env/bin/activate
#   pip install -U "huggingface_hub[cli]"
#   hf auth login         # paste your HF token (needs Llama-3.1 access approved)
#   hf download meta-llama/Llama-3.1-8B-Instruct --local-dir "/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/models/Llama-3.1-8B-Instruct"