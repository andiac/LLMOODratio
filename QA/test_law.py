from vllm import LLM, SamplingParams
import csv
import sys
import os

MAX_INT = sys.maxsize

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

# Apply the prompt template and system prompt of LLaMA-2-Chat demo for chat models (NOTE: NO prompt template is required for base models!)
our_system_prompt = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n" # Please do NOT change this

stop_tokens = []
llm = LLM(model="AdaptLLM/law-chat",tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=768, stop=stop_tokens)
# args = type('Args', (object, ), {'dataset': 'casehold'})
# ins = get_ins_casehold(args)

problem_prompt = '''You will be presented with a citing context from a legal decision and five potential holdings labeled from holding0 to holding4. Each holding statement is derived from the texts of legal citations. Please review the citing context and each holding carefully, then choose the label (holding0, holding1, holding2, holding3, or holding4) that you believe correctly represents the holding of the cited context. Additionally, explain the reasoning behind your selection. 
Citing context: {context}
holding0: {holding0}
holding1: {holding1}
holding2: {holding2}
holding3: {holding3}
holding4: {holding4}
'''
prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{problem_prompt} [/INST]"

ins = []
GTanswers = []
response = []
with open("./data/casehold_test.csv", "r") as f:
    reader = csv.reader(f)
    # skip the header
    next(reader, None)
    for row in reader:
        temp_instr = prompt.format(context=row[1], holding0=row[2], holding1=row[3], holding2=row[4], holding3=row[5], holding4=row[6]) # row[0] is the id
        ins.append(temp_instr)
        GTanswers.append(row[12])

batch_ins = batch_data(ins, batch_size=500)
for prompt in batch_ins:
    if isinstance(prompt, list):
        pass
    else:
        prompt = [prompt]
    completions = llm.generate(prompt, sampling_params)
    for output in completions:
        generated_text = output.outputs[0].text
        response.append(generated_text)

correct = []
for ans, pred in zip(GTanswers, response):
    if f"holding{ans}" in pred or f"Holding{ans}" in pred:
        correct.append(1)
    else:
        correct.append(0)

# save correct
os.makedirs("./results", exist_ok=True)
with open("./results/casehold_test_results.txt", "w") as f:
    for c in correct:
        f.write(f"{c}\n")

print(f"Accuracy: {sum(correct)/len(correct)}")

