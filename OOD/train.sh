#!/bin/bash

# Assume LOCAL is a boolean represented as a string ("true" or "false")
LOCAL="true" # Change this to "false" to test the other condition

if [ "$LOCAL" = "true" ]; then
    LLAMA="/root/autodl-tmp/Llama-2-7b-hf"
    MISTRAL="/root/autodl-tmp/Mistral-7B-v0.1"
else
    LLAMA="meta-llama/Llama-2-7b-hf"
    MISTRAL="mistralai/Mistral-7B-v0.1"
fi

# ROSTD Llama-2
python train.py --dataset ROSTD --max_length 384 --model_name $LLAMA --output_path ROSTD_Llama-2-7b-hf --lr 1e-4 --set_pad_id --train_batch_size 256 --eval_batch_size 256 --num_epochs 20
python test.py --dataset ROSTD --max_length 384 --model_name $LLAMA  --ckpt_path ROSTD_Llama-2-7b-hf/best_ckpt --dump_path ROSTD_Llama-2-7b-hf_epoch20.pkl --use_fp16

# ROSTD Mistral
python train.py --dataset ROSTD --max_length 384 --model_name $MISTRAL --set_pad_id --output_path ROSTD_Mistral-7B-v0.1 --lr 1e-4 --train_batch_size 256 --eval_batch_size 256 --num_epochs 10
python test.py --dataset ROSTD --max_length 384 --model_name $MISTRAL --ckpt_path ROSTD_Mistral-7B-v0.1/best_ckpt --dump_path  ROSTD_Mistral-7B-v0.1_epoch10.pkl --use_fp16

# SNIPS Llama
python train.py --dataset SNIPS --max_length 192 --model_name $LLAMA --output_path SNIPS_Llama-2-7b-hf --lr 1e-4 --set_pad_id --train_batch_size 256 --eval_batch_size 256 --num_epochs 20
python test.py --dataset SNIPS --max_length 192 --model_name $LLAMA --ckpt_path SNIPS_Llama-2-7b-hf/best_ckpt --dump_path SNIPS_Llama-2-7b-hf_epoch20.pkl --use_fp16

# SNIPS Mistral
python train.py --dataset SNIPS --max_length 192 --model_name $MISTRAL --set_pad_id --output_path SNIPS_Mistral-7B-v0.1 --lr 1e-4 --train_batch_size 384 --eval_batch_size 256 --num_epochs 20
python test.py --dataset SNIPS --max_length 192 --model_name $MISTRAL --ckpt_path SNIPS_Mistral-7B-v0.1/best_ckpt --dump_path SNIPS_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# CLINC150 Llama
python train.py --dataset CLINC --max_length 256 --model_name $LLAMA --set_pad_id --output_path CLINC_Llama-2-7b-hf --lr 1e-4 --train_batch_size 256 --eval_batch_size 256 --num_epochs 20 --lora_rank 32 --lora_alpha 64
python test.py --dataset CLINC --max_length 256 --model_name $LLAMA --ckpt_path CLINC_Llama-2-7b-hf/best_ckpt --dump_path CLINC_Llama-2-7b-hf_epoch20.pkl --use_fp16

# CLINC150 Mistral
python train.py --dataset CLINC --max_length 256 --model_name $MISTRAL --set_pad_id --output_path CLINC_Mistral-7B-v0.1 --lr 1e-4 --weight_decay 1e-4 --train_batch_size 256 --eval_batch_size 256 --num_epochs 10
python test.py --dataset CLINC --max_length 256 --model_name $MISTRAL --ckpt_path CLINC_Mistral-7B-v0.1/best_ckpt --dump_path CLINC_Mistral-7B-v0.1_epoch10.pkl --use_fp16

# 20NG
python train.py --dataset 20NG --max_length 256 --model_name $LLAMA --set_pad_id --output_path 20NG_Llama-2-7b-hf --lr 1e-4 --train_batch_size 32 --eval_batch_size 32 --num_epochs 20 --lora_rank 32 --lora_alpha 64
python train.py --dataset 20NG --max_length 256 --model_name $MISTRAL --set_pad_id --output_path 20NG_Mistral-7B-v0.1 --lr 1e-4 --train_batch_size 32 --eval_batch_size 32 --num_epochs 20 --lora_rank 32 --lora_alpha 64

# 20NG sst2
python test.py --dataset 20NG_sst2 --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_sst2_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_sst2 --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_sst2_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG mnli
python test.py --dataset 20NG_mnli --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_mnli_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_mnli --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_mnli_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG rte
python test.py --dataset 20NG_rte --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_rte_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_rte --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_rte_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG imdb
python test.py --dataset 20NG_imdb --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_imdb_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_imdb --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_imdb_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG multi30k
python test.py --dataset 20NG_multi30k --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_multi30k_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_multi30k --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_multi30k_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG NewsCategory
python test.py --dataset 20NG_news-category-hf --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_news-category_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_news-category-hf --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_news-category_Mistral-7B-v0.1_epoch20.pkl --use_fp16

# 20NG CLINC150
python test.py --dataset 20NG_clinc150 --max_length 256 --model_name $LLAMA --ckpt_path 20NG_Llama-2-7b-hf/best_ckpt --dump_path 20NG_clinc150_Llama-2-7b-hf_epoch20.pkl --use_fp16
python test.py --dataset 20NG_clinc150 --max_length 256 --model_name $MISTRAL --ckpt_path 20NG_Mistral-7B-v0.1/best_ckpt --dump_path 20NG_clinc150_Mistral-7B-v0.1_epoch20.pkl --use_fp16

