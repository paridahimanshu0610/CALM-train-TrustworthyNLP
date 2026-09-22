#!/bin/zsh
export raw_data='/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/data/train_val_data/CRA_resample_0.045M.json'
export conv_data='/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA_resample_0.045M_conv.json'
export data_name=CRA
export dev_data='/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA-resample-dev3k.json'
export train_data='/Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/data/CRA-resample-train4w.json'

# Ensure the output directory exists (creates parents as needed); no-op if it already exists
mkdir -p "$(dirname "${conv_data}")"

python /Users/himanshu/Documents/Projects/CALM-train-TrustworthyNLP/train/scripts/convert_to_conv_data.py \
    --orig_data ${raw_data} \
    --write_data ${conv_data} \
    --dataset_name CRA
head -n 3000 ${conv_data} > ${dev_data}
tail -n +3001 ${conv_data} > ${train_data}