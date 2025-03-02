import torch
from argparse import ArgumentParser
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import joblib
import datasets
from copy import deepcopy
from tqdm import tqdm

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

    arguments = parser.parse_args()
    return arguments



def get_likelihood(pretrained_model, finetuned_model, dataset, debug=False):
    likelihoods = []
    labels = []

    with torch.no_grad():
        for inputs in tqdm(dataset):
            # print(inputs)
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

            likelihoods.append((-pretrained_outputs.loss.item() * len(inputs['attention_mask']), -finetuned_outputs.loss.item() * len(inputs['attention_mask'])))
            labels.append(inputs['label'])

    return likelihoods, labels


def main(args):
    """
    Testing function
    """

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        is_decoder=True,
        torch_dtype=torch.float16,
    )
    pretrained_model.eval()

    if args.set_pad_id:
        pretrained_model.config.pad_token_id = model.config.eos_token_id

    finetuned_model = PeftModel.from_pretrained(deepcopy(pretrained_model), args.ckpt_path)
    finetuned_model.train(False)

    # move model to GPU device
    if pretrained_model.device.type != 'cuda':
        pretrained_model = pretrained_model.to('cuda:0')
    if finetuned_model.device.type != 'cuda':
        finetuned_model = finetuned_model.to('cuda:1')

    # load test data
    df_val = pd.read_csv(f'{args.dataset_name}_val.csv')
    df_test = pd.read_csv(f'{args.dataset_name}_test.csv')
    dataset = datasets.DatasetDict(
        {
            "val": datasets.Dataset.from_pandas(df_val),
            "test": datasets.Dataset.from_pandas(df_test),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        add_prefix_space=True,
    )

    if args.set_pad_id:
        tokenizer.pad_token = tokenizer.eos_token

    def _preprocesscing_function(examples):
        inputs = tokenizer(examples['text'], padding=False, truncation=True, max_length=args.max_length)
        inputs['label'] = examples['label']

        return inputs

    tokenized_datasets = dataset.map(_preprocesscing_function, batched=False)
    tokenized_datasets.set_format("torch")

    val_likelihoods, val_labels = get_likelihood(pretrained_model, finetuned_model, tokenized_datasets['val'])
    test_likelihoods, test_labels = get_likelihood(pretrained_model, finetuned_model, tokenized_datasets['test'])

    joblib.dump((val_likelihoods, val_labels, test_likelihoods, test_labels), args.dump_path)


if __name__ == "__main__":
    args = get_args()
    main(args)

