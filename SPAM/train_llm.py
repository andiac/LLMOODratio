import numpy as np
import random
import torch
import os
from argparse import ArgumentParser
from pathlib import Path
import datasets
from src.spamdetection.preprocessing import get_dataset, train_val_test_split
from src.spamdetection.preprocessing import init_datasets

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType

def set_seed(seed) -> None:
    """Fix random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        "--seed",
        type=int,
        default=0,
        required=False,
        help="Random seed",
    )
    # remove ham or spam
    parser.add_argument(
        "--indist",
        type=str,
        default=None,
        required=False,
        help="indist: ham or spam",
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
        "--lr", type=float, default=1e-4, help="Learning rate for training"
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

def remove_label(df, indist):
    if indist == "ham":
        new_df = df[df['label'] != 1]
    elif indist == "spam":
        new_df = df[df['label'] != 0]
    else:
        new_df = df
    new_df.reset_index()

    return new_df

args = get_args()
# makesure the split is the same
set_seed(args.seed)
assert args.indist in ["ham", "spam"], "indist must be either ham or spam"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_checkpoints = args.model_name
rank=args.lora_rank
alpha=args.lora_alpha
lora_dropout=args.lora_dropout
bias=args.lora_bias

# Download and process datasets
if os.path.exists("data") == False:
    init_datasets()

Path("outputs/csv").mkdir(parents=True, exist_ok=True)
Path("outputs/png").mkdir(parents=True, exist_ok=True)
Path("outputs/csv").mkdir(parents=True, exist_ok=True)

df = get_dataset(args.dataset_name)
(df_train, df_val, df_test), _ = train_val_test_split(
    df, train_size=0.8, has_val=True
)

# saved df_train
df_train.to_csv(f"./{args.dataset_name}_train.csv", index=False)
df_val.to_csv(f"./{args.dataset_name}_val.csv", index=False)
df_test.to_csv(f"./{args.dataset_name}_test.csv", index=False)

# Remove ham 
df_train = remove_label(df_train, args.indist)
df_val = remove_label(df_val, args.indist)
df_test = remove_label(df_test, args.indist)

dataset = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_pandas(df_train[['text']]),
        "val": datasets.Dataset.from_pandas(df_val[['text']]),
        "test": datasets.Dataset.from_pandas(df_test[['text']]),
    }
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
    eval_accumulation_steps=1,   # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/2
)

model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_checkpoints,
        load_in_8bit=False,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True,
        is_decoder=True,
    )
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
    # target_module = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoints,
    add_prefix_space=True,
)

if args.set_pad_id:
    tokenizer.pad_token = tokenizer.eos_token

def _preprocesscing_function(examples):
    return tokenizer(examples['text'], padding=False, truncation=True, max_length=args.max_length)

tokenized_datasets = dataset.map(_preprocesscing_function, batched=False)
tokenized_datasets.set_format("torch")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

if args.set_pad_id:
    model.config.pad_token_id = model.config.eos_token_id

# move model to GPU device
if model.device.type != 'cuda':
    model = model.to('cuda')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(os.path.join(args.output_path, 'best_ckpt'))

