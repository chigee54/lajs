#!/bin/bash

train_data_path=cail2022_类案检索_第二阶段/train/
dev_data_path=cail2022_类案检索_第一阶段/train/label_top30_dict.json

python train.py \
    --input $train_data_path \
    --dev_id_file $dev_data_path \
    --output_path ./saved \
    --model_path thunlp/Lawformer \
    --tokenizer_path hfl/chinese-roberta-wwm-ext \
    --max_length 1533


