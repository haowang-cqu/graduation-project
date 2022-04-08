#!/usr/bin/sh

DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 9 \
    --output_dir ./result/backdoor-wikitext-9-2 \
    --overwrite_output_dir \
    --save_strategy no \
    --overwrite_cache \
    --preprocessing_num_workers 8
