#!/bin/bash
#SBATCH --job-name=trustworthy_nlp_llama31_inference
#SBATCH --output=/scratch/user/paridahimanshu0610/trustworthy_nlp/CALM-train-TrustworthyNLP/inference/inference_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00

###################################
# Usage:
#   sbatch run_inference.sh FILE_TYPES=<val> SMOKE_TEST=<val> SMOKE_TEST_N=<val> \
#       MODELS=<val> LORA_CKPT_PATH=<val> LORA_MODEL_NAME=<val>
#
# Any argument may be omitted to use its default. Order doesn't matter.
#
#   FILE_TYPES      : test | bias | both   (default: both)
#   SMOKE_TEST      : true | false         (default: false)
#   SMOKE_TEST_N    : integer              (default: 5, only used if SMOKE_TEST=true)
#   MODELS          : base | lora | both   (default: both)
#   LORA_CKPT_PATH  : path to a checkpoint-XXXX dir. Required if MODELS is
#                     lora or both.
#   LORA_MODEL_NAME : optional. Output subfolder name for this LoRA run's
#                     results. If omitted, it's auto-derived from
#                     LORA_CKPT_PATH as "<run_dir_name>_<checkpoint_name>",
#                     e.g. "CRA-llama3.1-8b-instruct_CRA_0.045M_checkpoint-7010",
#                     so different fine-tuning runs never collide on output
#                     folders even if you forget to name them explicitly.
#
# Examples:
#   sbatch run_inference.sh FILE_TYPES=bias MODELS=base
#       # base model only
#
#   sbatch run_inference.sh FILE_TYPES=bias MODELS=lora \
#       LORA_CKPT_PATH=/scratch/.../saved_models/CRA-llama3.1-8b-instruct_CRA_0.045M/checkpoint-7010
#       # one LoRA checkpoint only, auto-named output folder
#
#   sbatch run_inference.sh FILE_TYPES=bias MODELS=both \
#       LORA_CKPT_PATH=/scratch/.../saved_models/CRA-llama3.1-8b-instruct_CRA_0.045M/checkpoint-7010
#       # both base and this LoRA checkpoint
#
#   sbatch run_inference.sh FILE_TYPES=bias SMOKE_TEST=true SMOKE_TEST_N=5 MODELS=lora \
#       LORA_CKPT_PATH=/scratch/.../saved_models/some_other_run/checkpoint-4200 \
#       LORA_MODEL_NAME=my_v2_run
#       # smoke test, LoRA only, explicit output folder name override
###################################

# Defaults
FILE_TYPES="both"
SMOKE_TEST="false"
SMOKE_TEST_N="5"
MODELS="both"
LORA_CKPT_PATH=""
LORA_MODEL_NAME=""

# Parse KEY=VALUE arguments (order-independent)
for arg in "$@"; do
    case "${arg}" in
        FILE_TYPES=*)      FILE_TYPES="${arg#*=}" ;;
        SMOKE_TEST=*)      SMOKE_TEST="${arg#*=}" ;;
        SMOKE_TEST_N=*)    SMOKE_TEST_N="${arg#*=}" ;;
        MODELS=*)          MODELS="${arg#*=}" ;;
        LORA_CKPT_PATH=*)  LORA_CKPT_PATH="${arg#*=}" ;;
        LORA_MODEL_NAME=*) LORA_MODEL_NAME="${arg#*=}" ;;
        *)
            echo "WARNING: unrecognized argument '${arg}', ignoring." >&2
            ;;
    esac
done

echo "Parsed arguments:"
echo "  FILE_TYPES=${FILE_TYPES}"
echo "  SMOKE_TEST=${SMOKE_TEST}"
echo "  SMOKE_TEST_N=${SMOKE_TEST_N}"
echo "  MODELS=${MODELS}"
echo "  LORA_CKPT_PATH=${LORA_CKPT_PATH}"
echo "  LORA_MODEL_NAME=${LORA_MODEL_NAME}"

if [ "${MODELS}" = "lora" ] || [ "${MODELS}" = "both" ]; then
    if [ -z "${LORA_CKPT_PATH}" ]; then
        echo "ERROR: MODELS=${MODELS} requires LORA_CKPT_PATH=<path> to be set." >&2
        exit 1
    fi
    if [ -z "${LORA_MODEL_NAME}" ]; then
        run_name=$(basename "$(dirname "${LORA_CKPT_PATH}")")
        ckpt_name=$(basename "${LORA_CKPT_PATH}")
        LORA_MODEL_NAME="${run_name}_${ckpt_name}"
        echo "  (auto-derived LORA_MODEL_NAME=${LORA_MODEL_NAME})"
    fi
fi

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

echo "Base Model: ${model_name_or_path}"

###################################
# Build optional smoke-test flags
###################################
SMOKE_FLAGS=""
if [ "${SMOKE_TEST}" = "true" ]; then
    SMOKE_FLAGS="--smoke_test --smoke_test_n ${SMOKE_TEST_N}"
fi

###################################
# Run inference: base model
###################################
if [ "${MODELS}" = "base" ] || [ "${MODELS}" = "both" ]; then
    echo "=== Running base model inference ==="
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
        > "${inference_dir}/inference_base_${FILE_TYPES}_${SMOKE_TEST}.log" 2>&1

    echo "Base model inference completed!"
fi

###################################
# Run inference: LoRA fine-tuned checkpoint
###################################
if [ "${MODELS}" = "lora" ] || [ "${MODELS}" = "both" ]; then
    echo "=== Running LoRA checkpoint inference (${LORA_CKPT_PATH}) ==="
    echo "Output folder name: ${LORA_MODEL_NAME}"
    python "${ABS_PATH}/train/src/entry_point/inference.py" \
        --model_name_or_path "${model_name_or_path}" \
        --ckpt_path "${LORA_CKPT_PATH}" \
        --mode lora \
        --llama \
        --model_name "${LORA_MODEL_NAME}" \
        --file_types "${FILE_TYPES}" \
        --query_key_test chat_query \
        --query_key_bias normal_query \
        --batch_size 8 \
        ${SMOKE_FLAGS} \
        > "${inference_dir}/inference_lora_${LORA_MODEL_NAME}_${FILE_TYPES}_${SMOKE_TEST}.log" 2>&1

    echo "LoRA checkpoint inference completed!"
fi

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
#   huggingface-cli login          # paste your HF token (needs Llama-3.1 access approved)
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
#       --local-dir "$ABS_PATH/models/Llama-3.1-8B-Instruct"