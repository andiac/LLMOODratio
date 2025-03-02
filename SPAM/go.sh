#!/bin/bash

datasets=("ling" "sms" "spamassassin" "enron")

for dataset in "${datasets[@]}"; do
    # Training for ham
    python train_llm.py --dataset $dataset --max_length 384 --model_name meta-llama/Llama-2-7b-hf --output_path ${dataset}_ham_Llama-2-7b-hf --set_pad_id --train_batch_size 64 --eval_batch_size 32 --num_epochs 20 --indist ham
    python test_llm.py --dataset $dataset --max_length 384 --model_name meta-llama/Llama-2-7b-hf  --ckpt_path ${dataset}_ham_Llama-2-7b-hf/best_ckpt --dump_path ${dataset}_ham_Llama-2-7b-hf_epoch20.pkl
    python eval_llm.py --dump_path ${dataset}_ham_Llama-2-7b-hf_epoch20.pkl --indist ham --metric lr 
    python eval_llm.py --dump_path ${dataset}_ham_Llama-2-7b-hf_epoch20.pkl --indist ham --metric lh 

    # Training for spam
    python train_llm.py --dataset $dataset --max_length 384 --model_name meta-llama/Llama-2-7b-hf --output_path ${dataset}_spam_Llama-2-7b-hf --set_pad_id --train_batch_size 64 --eval_batch_size 32 --num_epochs 20 --indist spam
    python test_llm.py --dataset $dataset --max_length 384 --model_name meta-llama/Llama-2-7b-hf  --ckpt_path ${dataset}_spam_Llama-2-7b-hf/best_ckpt --dump_path ${dataset}_spam_Llama-2-7b-hf_epoch20.pkl
    python eval_llm.py --dump_path ${dataset}_spam_Llama-2-7b-hf_epoch20.pkl --indist spam --metric lr 
    python eval_llm.py --dump_path ${dataset}_spam_Llama-2-7b-hf_epoch20.pkl --indist spam --metric lh

    # Double feature testing and evaluation
    python test_llm_double_ft.py --dataset $dataset --max_length 384 --model_name meta-llama/Llama-2-7b-hf  --spam_path ${dataset}_spam_Llama-2-7b-hf/best_ckpt --ham_path ${dataset}_ham_Llama-2-7b-hf/best_ckpt --dump_path ${dataset}_double_Llama-2-7b-hf_epoch20.pkl
    python eval_llm.py --dump_path ${dataset}_double_Llama-2-7b-hf_epoch20.pkl --indist spam --metric lr 
    python eval_llm.py --dump_path ${dataset}_double_Llama-2-7b-hf_epoch20.pkl --indist spam --metric lh
done

