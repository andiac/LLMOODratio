import argparse
import json
import csv
import re
import os
import jsonlines
from vllm import LLM, SamplingParams
import sys

MAX_INT = sys.maxsize

# Apply the prompt template and system prompt of LLaMA-2-Chat demo for chat models (NOTE: NO prompt template is required for base models!)
our_system_prompt = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n" # Please do NOT change this


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

# noc means no context
dataset_dict = {"GSM8K": "./data/GSM8K_test.jsonl", "MATH": "./data/MATH_test.jsonl", "boolq": "./data/boolq_dev.jsonl", "SQUAD": "./data/squad_v2_dev.json", "PIQA": "./data/piqa_valid.jsonl", "casehold": "./data/casehold_test.csv", "FIQA": "./data/FiQA_test_question_task2.tsv", "PubMedQA": "./data/PubMedQA_test_set.json", "FPB": "./data/FPB_Sentences_AllAgree.txt"}

def get_ins_SQUAD(args):
    ins = []
    problem_prompt = "{context}\n{question}\n"
    # problem_prompt = "{question}\n"
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r") as f:
        data_dict = json.load(f)
        for d in data_dict['data']:
            for p in d['paragraphs']:
                for q in p['qas']:
                    temp_instr = prompt.format(context=p['context'], question=q['question'])
                    ins.append(temp_instr)
    return ins

def get_ins_GSM8K(args):
    ins = []
    problem_prompt = "{instruction}\n"
    # problem_prompt = (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    # )
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = prompt.format(instruction=item["query"])
            ins.append(temp_instr)
    return ins

def get_ins_MATH(args):
    ins = []
    problem_prompt = "{instruction}\n"
    # problem_prompt = (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    # )
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = prompt.format(instruction=item["instruction"])
            ins.append(temp_instr)
    return ins

def get_ins_boolq(args):
    ins = []
    problem_prompt = "{context}\n{question}?\n"
    # problem_prompt = "{question}?\n"
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = prompt.format(context=item['passage'], question=item['question'])
            ins.append(temp_instr)
    return ins

def get_ins_PIQA(args):
    ins = []
    problem_prompt = '''You will be presented with a question followed by two potential solutions labeled as sol1 and sol2. 
question: {question}
sol1: {sol1}
sol2: {sol2}
Please select and answer with the label (either sol1 or sol2) that you believe correctly solves the question provided.
'''
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = prompt.format(question=item['goal'], sol1=item['sol1'], sol2=item['sol2'])
            ins.append(temp_instr)
    return ins

def get_ins_casehold(args):
    ins = []
    problem_prompt = '''You will be presented with a citing context from a legal decision and five potential holdings labeled from holding0 to holding4. Each holding statement is derived from the texts of legal citations. Please review the citing context and each holding carefully, then choose the label (holding0, holding1, holding2, holding3, or holding4) that you believe correctly represents the holding of the cited context. Additionally, explain the reasoning behind your selection. 
Citing context: {context}
holding0: {holding0}
holding1: {holding1}
holding2: {holding2}
holding3: {holding3}
holding4: {holding4}
'''
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r") as f:
        reader = csv.reader(f)
        # skip the header
        next(reader, None)
        for row in reader:
            temp_instr = prompt.format(context=row[1], holding0=row[2], holding1=row[3], holding2=row[4], holding3=row[5], holding4=row[6]) # row[0] is the id
            ins.append(temp_instr)
    return ins

def get_ins_FIQA(args):
    ins = []
    problem_prompt = '''
    {question}
'''
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    with open(dataset_dict[args.dataset], "r") as f:
        reader = csv.reader(f, delimiter='\t')
        # skip the header
        next(reader, None)
        for row in reader:
            temp_instr = prompt.format(question=row[2])
            ins.append(temp_instr)
    return ins

def get_ins_PubMedQA(args):
    ins = []
    problem_prompt = "{context}\n{question}\nPlease begin your response with 'Yes,' 'No,' or 'Maybe' when answering the question. Please explain the reason for your choice.\n"
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    
    with open(dataset_dict[args.dataset], "r") as f:
        data_dict = json.load(f)
        for item in data_dict.values():
            temp_instr = prompt.format(context=" ".join(item['CONTEXTS']), question=item['QUESTION'])
            ins.append(temp_instr)

    return ins

def get_ins_FPB(args):
    ins = []
    problem_prompt = '''You will be given a sentence and asked to classify its sentiment as either 'positive,' 'neutral,' or 'negative.' Please select the sentiment that best matches the sentence and provide an explanation for your choice.

Sentence: {context} 
'''
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"
    
    with open(dataset_dict[args.dataset], "r+", encoding="latin-1") as f:
        # it is a text file, load it line by line
        for line in f:
            word_list = line.strip().split()[:-1]
            sentence = " ".join(word_list)
            temp_instr = prompt.format(context=sentence)
            ins.append(temp_instr)

    return ins

func_dict = {"GSM8K": get_ins_GSM8K, "MATH": get_ins_MATH, "boolq": get_ins_boolq, "SQUAD": get_ins_SQUAD, "PIQA": get_ins_PIQA, "casehold": get_ins_casehold, "FIQA": get_ins_FIQA, "PubMedQA": get_ins_PubMedQA, "FPB": get_ins_FPB}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--dataset", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    args = parser.parse_args()

    instructions = func_dict[args.dataset](args)
    instructions = instructions[args.start:args.end]
    batch_ins    = batch_data(instructions, batch_size=args.batch_size)

    stop_tokens = []
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=768, stop=stop_tokens)
    llm = LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size)
    answers = []

    for idx, prompt in enumerate(batch_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            answers.append(generated_text)

    os.makedirs(f"./samples/{os.path.basename(args.model)}", exist_ok=True)
    with jsonlines.open(os.path.join(f"./samples/{os.path.basename(args.model)}", f"{args.dataset}.jsonl"), mode='w') as writer:
        for question, answer in zip(instructions, answers):
            writer.write({"question": question, "answer": answer})

