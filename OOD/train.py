r"""
Training script to fine-tune a pre-train LLM with PEFT methods using HuggingFace.
  Example to run this script:
    python train.py \
     --dataset ROSTD \
     --max_length 384 \
     --model_name roberta-large \
     --output_path ROSTD_roberta-large \
     --train_batch_size 512 \
     --eval_batch_size 256
"""

import os
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, TrainerCallback
import pdb
from dataset_utils import *
import warnings
warnings.filterwarnings('ignore')


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["WANDB_MODE"] = "dryrun"


def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to store the fine-tuned model",
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
        "--lr", type=float, default=1e-3, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            #self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval")
            return control_copy


def get_dataset_and_collator(
    dataset_name,
    model_checkpoints,
    add_prefix_space=True,
    max_length=512,
    truncation=True,
    set_pad_id=False,
    mode='train',
):
    """
    Load the specified dataset

    Paramters:
    ---------
    dataset_name: str
        Name of the dataset
    model_checkpoints:
        Name of the pre-trained model to use for tokenization
    """
    if dataset_name == 'ROSTD':
        # 使用pandas读取CSV文件
        train_df = pd.read_csv('dataset/fbrelease/unsup/OODRemovedtrain.tsv', delimiter='\t', names=['category', 'text', 'tag'])
        val_df = pd.read_csv('dataset/fbrelease/unsup/eval.tsv', delimiter='\t', names=['category', 'text', 'tag'])
        test_df = pd.read_csv('dataset/fbrelease/unsup/test.tsv', delimiter='\t', names=['category', 'text', 'tag'])

        def remove_ood(df):
            new_df = df[df['tag'] != 'FILLER']
            new_df.reset_index()

            return new_df

        def add_label(df):
            df['label'] = df['tag'] == 'FILLER'
            df.reset_index()

            return df

    elif dataset_name == 'SNIPS':
        # load csv
        train_df = pd.read_csv('dataset/snips/train.csv')
        val_df = pd.read_csv('dataset/snips/valid.csv')
        test_df = pd.read_csv('dataset/snips/test.csv')

        def remove_ood(df):
            new_df = df[(df['label'] != 'GetWeather') & (df['label'] != 'BookRestaurant')]
            new_df.reset_index()

            return new_df

        def add_label(df):
            df['label'] = (df['label'] == 'GetWeather') | (df['label'] == 'BookRestaurant')
            df.reset_index()

            return df

    elif dataset_name == 'CLINC':
        data_util = DatasetUtil('clinc150')
        dataset = data_util.get_dataset('clinc150', split=None)

        train_df = data_util.merge_dataset_splits([dataset['train'], dataset['oos_train']]).to_pandas()
        train_df.rename(columns={'sentence':'text'}, inplace=True)
        val_df = data_util.merge_dataset_splits([dataset['val'], dataset['oos_val']]).to_pandas()
        val_df.rename(columns={'sentence':'text'}, inplace=True)
        test_df = data_util.merge_dataset_splits([dataset['test'], dataset['oos_test']]).to_pandas()
        test_df.rename(columns={'sentence':'text'}, inplace=True)

        def remove_ood(df):
            new_df = df[df['label'] != 150]
            new_df.reset_index()

            return new_df

        def add_label(df):
            df['label'] = df['label'] == 150
            df.reset_index()

            return df

    elif '20NG' == dataset_name[:4]:
        data_util = DatasetUtil('20NG')
        dataset = data_util.get_dataset('20newsgroups', split=None)

        train_df = dataset['train'].to_pandas()
        train_df.rename(columns={'sentence':'text'}, inplace=True)

        def remove_ood(df):
            return df

        val_df = dataset['test'].to_pandas()
        val_df.rename(columns={'sentence':'text'}, inplace=True)
        test_df = val_df

        # sst2, mnli, rte, imdb, multi30k, news-category-hf, clinc150
        if '_' in dataset_name:
            ood_dataset_name = dataset_name.split('_')[-1]
            ood_test_dataset = data_util.get_dataset(ood_dataset_name, split=None)
            print(ood_test_dataset)
            if ood_dataset_name == 'clinc150':
                ood_test_df = ood_test_dataset['oos_test'].to_pandas()
            else:
                ood_test_df = ood_test_dataset['test'].to_pandas()

            if ood_dataset_name == 'imdb':
                pass
            elif ood_dataset_name == 'mnli':
                ood_test_df['text'] = ood_test_df['premise'] + ood_test_df['hypothesis']
            elif ood_dataset_name == 'rte':
                ood_test_df['text'] = ood_test_df['sentence1'] + ood_test_df['sentence2']
            elif ood_dataset_name == 'news-category-hf':
                ood_test_df['text'] = ood_test_df['headline'] + ood_test_df['short_description']
            else:
                ood_test_df.rename(columns={'sentence':'text'}, inplace=True)
            test_df['label'] = 0
            ood_test_df['label'] = 1
            test_df = pd.concat([test_df[['text', 'label']], ood_test_df[['text', 'label']]])
            val_df = test_df

            def add_label(df):
                return df

    elif 'imdb' == dataset_name[:4]:
        data_util = DatasetUtil('imdb')
        dataset = data_util.get_dataset('imdb', split=None)

        def remove_ood(df):
            return df

        print(dataset)
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        val_df = test_df
        print(train_df)
        print(val_df)

        # sst2
        if '_' in dataset_name:
            ood_dataset_name = dataset_name.split('_')[-1]
            ood_test_dataset = data_util.get_dataset(ood_dataset_name, split=None)
            ood_test_df = ood_test_dataset['test'].to_pandas()
            ood_test_df.rename(columns={'sentence':'text'}, inplace=True)
            test_df['label'] = 0
            ood_test_df['label'] = 1
            test_df = pd.concat([test_df[['text', 'label']], ood_test_df[['text', 'label']]])
            val_df = test_df

            def add_label(df):
                return df

    train_max_length = max(len(text.split()) for text in train_df['text'])
    val_max_length = max(len(text.split()) for text in val_df['text'])
    test_max_length = max(len(text.split()) for text in test_df['text'])

    print("[Before remove_ood] Train max length:", train_max_length)
    print("[Before remove_ood] Val max length:", val_max_length)
    print("[Before remove_ood] Test max length:", test_max_length)

    if mode == 'train':
        train_df = remove_ood(train_df)
        val_df = remove_ood(val_df)
        test_df = remove_ood(test_df)

        train_max_length = max(len(text.split()) for text in train_df['text'])
        val_max_length = max(len(text.split()) for text in val_df['text'])
        test_max_length = max(len(text.split()) for text in test_df['text'])

        print("[After remove_ood] Train max length:", train_max_length)
        print("[After remove_ood] Val max length:", val_max_length)
        print("[After remove_ood] Test max length:", test_max_length)
    else:
        train_df = add_label(train_df)
        val_df = add_label(val_df)
        test_df = add_label(test_df)

    if mode == 'train':
        train_dataset = Dataset.from_pandas(train_df[['text']])
        val_dataset = Dataset.from_pandas(val_df[['text']])
        test_dataset = Dataset.from_pandas(test_df[['text']])
    else:
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    data = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })


    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints,
        add_prefix_space=add_prefix_space
    )

    if set_pad_id:
        tokenizer.pad_token = tokenizer.eos_token

    def _preprocesscing_function(examples):
        #return tokenizer(examples['text'], padding='longest', truncation=truncation, max_length=max_length)
        return tokenizer(examples['text'], padding=False, truncation=truncation, max_length=max_length)

    tokenized_datasets = data.map(_preprocesscing_function, batched=False)
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenized_datasets, data_collator


def get_lora_model(model_checkpoints, rank=16, alpha=32, lora_dropout=0.05, bias='none'):
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoints,
            load_in_8bit=False,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True,
            is_decoder=True,
        )
    if model_checkpoints == 'mistralai/Mistral-7B-v0.1':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )

    if model_checkpoints == 'meta-llama/Llama-2-7b-hf':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )

    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def main(args):
    """
    Training function
    """
    dataset, collator = get_dataset_and_collator(
        args.dataset_name,
        args.model_name,
        max_length=args.max_length,
        set_pad_id=args.set_pad_id,
        add_prefix_space=True,
        truncation=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type= "cosine",
        warmup_ratio= 0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=True,
        max_grad_norm= 0.1,
    )

    model = get_lora_model(
        args.model_name,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )
    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset["val"],
        data_collator=collator,
    )
    trainer.add_callback(CustomCallback(trainer))
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(os.path.join(args.output_path, 'best_ckpt'))


if __name__ == "__main__":
    args = get_args()
    main(args)

