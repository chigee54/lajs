#!/bin/bash

data_path=cail2022_类案检索_封测阶段/

python predict.py \
    --input $data_path \
    --output ./ \
    --encoder_path saved/bsz1_lr1e-05/lawformer_best.pt \
    --rank_mlp_path saved/bsz1_lr5e-05/rank_mlp_best.pt \
    --model_path thunlp/Lawformer \
    --tokenizer_path hfl/chinese-roberta-wwm-ext \
    --extract_batch_size 10 \
    --max_length 1533


