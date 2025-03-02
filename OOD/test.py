r"""
Testing script for OOD detection.
  Example to run this script:
    python test.py \
     --dataset_name ROSTD \
     --max_length 384 \
     --model_name roberta-large \
     --ckpt_path ROSTD_roberta-large-epoch5/best_ckpt \
     --dump_path ROSTD_roberta-large.pkl
"""

import os
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import pdb
import joblib
from tqdm import tqdm
from train import get_dataset_and_collator
import warnings
warnings.filterwarnings('ignore')


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_args():
    parser = ArgumentParser(description="OOD detection")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        required=True,
        help="Path to stored checkpoint",
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default=None,
        required=True,
        help="Path to store (val_likelihoods, val_labels, test_likelihoods, test_labels)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id",
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--evaluate_val",
        default=False,
        action='store_true',
        help="wether to evaluate val datatset",
    )
    parser.add_argument(
        "--use_fp16",
        default=False,
        action='store_true',
        help="wether to use fp16 for inference",
    )

    arguments = parser.parse_args()
    return arguments


def get_likelihood(pretrained_model, finetuned_model, dataset, debug=False):
    likelihoods = []
    labels = []

    with torch.no_grad():
        for inputs in tqdm(dataset):
            pretrained_outputs = pretrained_model(
                input_ids=inputs['input_ids'].unsqueeze(0).to('cuda:0'),
                attention_mask=inputs['attention_mask'].unsqueeze(0).to('cuda:0'),
                labels=inputs["input_ids"].unsqueeze(0).to('cuda:0')
            )
            finetuned_outputs = finetuned_model(
                input_ids=inputs['input_ids'].unsqueeze(0).to('cuda:1'),
                attention_mask=inputs['attention_mask'].unsqueeze(0).to('cuda:1'),
                labels=inputs["input_ids"].unsqueeze(0).to('cuda:1')
            )

            likelihoods.append((-pretrained_outputs.loss.item(),
                                -finetuned_outputs.loss.item(),
                                -pretrained_outputs.loss.item() * len(inputs['input_ids']),
                                -finetuned_outputs.loss.item() * len(inputs['input_ids'])))
            labels.append(inputs['label'])

            # 打印loss(likelihood即为负的loss)
            if debug:
                print(f'pretrained loss: {pretrained_outputs.loss.item()}')
                print(f'finetuned loss: {finetuned_outputs.loss.item()}')

    return likelihoods, labels


def main(args):
    """
    Testing function
    """
    dataset, _ = get_dataset_and_collator(
        args.dataset_name,
        args.model_name,
        max_length=args.max_length,
        set_pad_id=args.set_pad_id,
        add_prefix_space=True,
        truncation=True,
        mode='test'
    )

    val_likelihoods = []
    val_labels = []
    test_likelihoods = []
    test_labels = []

    dtypes = []
    if args.use_fp16:
        dtypes.append(torch.float16)
    else:
        dtypes.append(torch.float32)
    for dtype in dtypes:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            is_decoder=True,
            torch_dtype=dtype,
        )

        finetuned_model = PeftModel.from_pretrained(deepcopy(pretrained_model), args.ckpt_path)
        finetuned_model.train(False)

        # move model to GPU device
        if pretrained_model.device.type != 'cuda':
            pretrained_model = pretrained_model.to('cuda:0')
        if finetuned_model.device.type != 'cuda':
            finetuned_model = finetuned_model.to('cuda:1')

        if args.evaluate_val:
            cur_val_likelihoods, val_labels = get_likelihood(pretrained_model, finetuned_model, dataset['val'])
            val_likelihoods.append(cur_val_likelihoods)

        cur_test_likelihoods, test_labels = get_likelihood(pretrained_model, finetuned_model, dataset['test'])
        test_likelihoods.append(cur_test_likelihoods)

    joblib.dump((val_likelihoods, val_labels, test_likelihoods, test_labels), args.dump_path)


if __name__ == "__main__":
    args = get_args()
    main(args)

