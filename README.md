## Project Structure

Below are the key components of the project:

### Training Logs
- `train/train_log_dir/debias_finetuning_train.log` — Debiased CALM fine-tuning log  
- `train/train_log_dir/train.log` — Standard CALM fine-tuning log

### Saved Models
- `train/saved_models/CRA-llama2-7b-chat_CRA_0.045M` — Saved model checkpoint for CALM  
- `train/saved_models/CRA-llama2-7b-chat_CRA_debiased` — Saved model checkpoint for Debiased CALM

### Bias & Performance Evaluation
- `bias_testing/` — Scripts to compute performance metrics and bias evaluation across datasets.

### Data
- `data/original_data/` — Datasets for all four tasks.  
  - See `preprocess.py` for the main data generation script.
- `data/split_data/` — Inference and evaluation datasets.

### Inference
- `inference/model_inference/` — Contains inference outputs for all datasets across models.  
  - Folder names indicate the model used.

### Models
- `models/Llama-2-7b-chat-hf/` — Base model used for fine-tuning.

### Training & Inference Scripts
- `train/scripts/run_sft.sh` — SLURM script for training.
- `train/scripts/run_inference.sh` — SLURM script for inference.

### Entry Points
- `train/src/entry_point/mac_inference.py` — Script for local inference.
- `train/src/entry_point/inference.py` — Common script for model-wise inference.
