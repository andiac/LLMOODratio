import pickle

with open(f'./likelihoods/law-chat/in_casehold_a.pkl', 'rb') as f:
    in_casehold_likelihood = pickle.load(f)

with open(f'./likelihoods/law-chat/out_casehold_a.pkl', 'rb') as f:
    out_casehold_likelihood = pickle.load(f)

casehold_ratio = [o - i for i, o in zip(in_casehold_likelihood, out_casehold_likelihood)]


with open(f'./likelihoods/law-chat/in_MATH_a.pkl', 'rb') as f:
    in_MATH_likelihood = pickle.load(f)

with open(f'./likelihoods/law-chat/out_MATH_a.pkl', 'rb') as f:
    out_MATH_likelihood = pickle.load(f)

MATH_ratio = [o - i for i, o in zip(in_MATH_likelihood, out_MATH_likelihood)]

casehold_correct = []
with open("./results/casehold_test_results.txt", "r") as f:
    for line in f:
        if int(line.strip()) == 1:
            casehold_correct.append(True)
        else:
            casehold_correct.append(False)

assert len(casehold_correct) == len(casehold_ratio)

MATH_correct = [False] * len(MATH_ratio)

casehold_ratio, casehold_correct = zip(*sorted(zip(casehold_ratio, casehold_correct)))
MATH_ratio, MATH_correct = zip(*sorted(zip(MATH_ratio, MATH_correct)))

# old correct
print("casehold correct:")
print(sum(casehold_correct) / len(casehold_correct))

# sum correct
print("sum correct:")
print((sum(casehold_correct)) / (len(casehold_correct) + len(MATH_correct)))

# get 90% quantile of casehold_ratio
print("90% quantile:")
casehold_ratio_90 = casehold_ratio[int(len(casehold_ratio) * 0.9)]
casehold_correct_90 = casehold_correct[0:int(len(casehold_correct) * 0.9)]
print(casehold_ratio_90)
print(type(casehold_correct_90))

print(sum(casehold_correct_90) / len(casehold_correct_90))

MATH_remain_90 = [r for r in MATH_ratio if r < casehold_ratio_90]
print(len(MATH_remain_90))
new_correct = list(casehold_correct_90) + [False] * len(MATH_remain_90)

print(sum(new_correct) / len(new_correct))

# get 100% quantile of casehold_ratio
print("100% quantile:")
casehold_ratio_100 = casehold_ratio[-1]
casehold_correct_100 = casehold_correct
print(casehold_ratio_100)
print(type(casehold_correct_100))

print(sum(casehold_correct_100) / len(casehold_correct_100))

MATH_remain_100 = [r for r in MATH_ratio if r < casehold_ratio_100]
print(len(MATH_remain_100))
new_correct = list(casehold_correct_100) + [False] * len(MATH_remain_100)

print(sum(new_correct) / len(new_correct))

# get 95% quantile of casehold_ratio
print("95% quantile:")
casehold_ratio_95 = casehold_ratio[int(len(casehold_ratio) * 0.95)]
casehold_correct_95 = casehold_correct[0:int(len(casehold_correct) * 0.95)]
print(casehold_ratio_95)
print(type(casehold_correct_95))

print(sum(casehold_correct_95) / len(casehold_correct_95))

MATH_remain_95 = [r for r in MATH_ratio if r < casehold_ratio_95]
print(len(MATH_remain_95))
new_correct = list(casehold_correct_95) + [False] * len(MATH_remain_95)

print(sum(new_correct) / len(new_correct))

# get 85% quantile of casehold_ratio
print("85% quantile:")
casehold_ratio_85 = casehold_ratio[int(len(casehold_ratio) * 0.85)]
casehold_correct_85 = casehold_correct[0:int(len(casehold_correct) * 0.85)]
print(casehold_ratio_85)
print(type(casehold_correct_85))

print(sum(casehold_correct_85) / len(casehold_correct_85))

MATH_remain_85 = [r for r in MATH_ratio if r < casehold_ratio_85]
print(len(MATH_remain_85))
new_correct = list(casehold_correct_85) + [False] * len(MATH_remain_85)

print(sum(new_correct) / len(new_correct))

# get 80% quantile of casehold_ratio
print("80% quantile:")
casehold_ratio_80 = casehold_ratio[int(len(casehold_ratio) * 0.8)]
casehold_correct_80 = casehold_correct[0:int(len(casehold_correct) * 0.8)]
print(casehold_ratio_80)
print(type(casehold_correct_80))

print(sum(casehold_correct_80) / len(casehold_correct_80))

MATH_remain_80 = [r for r in MATH_ratio if r < casehold_ratio_80]
print(len(MATH_remain_80))
new_correct = list(casehold_correct_80) + [False] * len(MATH_remain_80)

print(sum(new_correct) / len(new_correct))

# get 75% quantile of casehold_ratio
print("75% quantile:")
casehold_ratio_75 = casehold_ratio[int(len(casehold_ratio) * 0.75)]
casehold_correct_75 = casehold_correct[0:int(len(casehold_correct) * 0.75)]
print(casehold_ratio_75)
print(type(casehold_correct_75))

print(sum(casehold_correct_75) / len(casehold_correct_75))

MATH_remain_75 = [r for r in MATH_ratio if r < casehold_ratio_75]
print(len(MATH_remain_75))
new_correct = list(casehold_correct_75) + [False] * len(MATH_remain_75)

print(sum(new_correct) / len(new_correct))

# get 50% quantile of casehold_ratio
print("50% quantile:")
casehold_ratio_50 = casehold_ratio[int(len(casehold_ratio) * 0.5)]
casehold_correct_50 = casehold_correct[0:int(len(casehold_correct) * 0.5)]
print(casehold_ratio_50)
print(type(casehold_correct_50))

print(sum(casehold_correct_50) / len(casehold_correct_50))

MATH_remain_50 = [r for r in MATH_ratio if r < casehold_ratio_50]
print(len(MATH_remain_50))
new_correct = list(casehold_correct_50) + [False] * len(MATH_remain_50)

print(sum(new_correct) / len(new_correct))

# get 25% quantile of casehold_ratio
print("25% quantile:")
casehold_ratio_25 = casehold_ratio[int(len(casehold_ratio) * 0.25)]
casehold_correct_25 = casehold_correct[0:int(len(casehold_correct) * 0.25)]
print(casehold_ratio_25)
print(type(casehold_correct_25))

print(sum(casehold_correct_25) / len(casehold_correct_25))

MATH_remain_25 = [r for r in MATH_ratio if r < casehold_ratio_25]
print(len(MATH_remain_25))
new_correct = list(casehold_correct_25) + [False] * len(MATH_remain_25)

print(sum(new_correct) / len(new_correct))


