import torch
import argparse
import jsonlines
import os
import pickle
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

parser = argparse.ArgumentParser()
parser.add_argument("--in_model", type=str, default='/root/autodl-tmp/MetaMath-7B-V1.0')
parser.add_argument("--out_model", type=str, default='/root/autodl-tmp/MetaMath-7B-V1.0')
parser.add_argument("--dataset", type=str, default='')
parser.add_argument("--form", type=str, default='')
parser.add_argument("--mode", type=str, default='in')
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    required=False,
    help="Maximum length of the input sequences",
)
args = parser.parse_args()
assert args.form in ["q", "a", "qa"]

pretrained_model = AutoModelForCausalLM.from_pretrained(
    args.in_model if args.mode == "in" else args.out_model,
    is_decoder=True,
    torch_dtype=torch.float16,
)
pretrained_model = pretrained_model.to('cuda:0')

sentences = []
sample_file_path = f"./samples/{os.path.basename(args.in_model)}/{args.dataset}.jsonl"
with open(sample_file_path, "r+", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        if args.form == "q":
            sentences.append("{question}\n".format(question=item["question"]))
        elif args.form == "a":
            sentences.append("{answer}\n".format(answer=item["answer"]))
        # qa
        else:
            sentences.append("{question}\n{answer}\n".format(question=item["question"], answer=item["answer"]))

sentences_dict = {"sentences": sentences}
sentences_data = Dataset.from_dict(sentences_dict)

tokenizer = AutoTokenizer.from_pretrained(
    args.in_model if args.mode == "in" else args.out_model,
    add_prefix_space=True
)
def _preprocesscing_function(examples):
    return tokenizer(examples['sentences'], padding=False, truncation=True, max_length=args.max_length)

tokenized_dataset = sentences_data.map(_preprocesscing_function, batched=False)
tokenized_dataset.set_format("torch")

loglikelihoods = []
with torch.no_grad():
    for inputs in tqdm(tokenized_dataset):
        outputs = pretrained_model(
            input_ids=inputs['input_ids'].unsqueeze(0).to('cuda:0'),
            attention_mask=inputs['attention_mask'].unsqueeze(0).to('cuda:0'),
            labels=inputs["input_ids"].unsqueeze(0).to('cuda:0')
        )
        loglikelihoods.append(-outputs.loss.item() * len(inputs['input_ids']))
        # loglikelihoods.append(-outputs.loss.item())

os.makedirs(f"./likelihoods/{os.path.basename(args.in_model)}", exist_ok=True)
with open(f'./likelihoods/{os.path.basename(args.in_model)}/{args.mode}_{args.dataset}_{args.form}.pkl', 'wb') as f:
    pickle.dump(loglikelihoods, f)

