import os
import argparse
from ood_metrics import calc_metrics
import pickle

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--in_model", type=str, default='/root/autodl-tmp/MetaMath-7B-V1.0')
parser.add_argument("--out_model", type=str, default='/root/autodl-tmp/MetaMath-7B-V1.0')
parser.add_argument("--in_dist", type=str, default='GSM8K')   # in-dist name
parser.add_argument("--out_dist", type=str, default='SQUAD')  # ood name
parser.add_argument("--mode", type=str, default='qa')
args = parser.parse_args()

assert args.mode in ['q', 'a', 'qa', 'agivenq']

gt_ood = [] # ground truth 

if args.mode == 'q' or args.mode == 'a' or args.mode == 'qa':
    in_lh = []
    out_lh    = []
    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.in_dist}_{args.mode}.pkl', 'rb') as f:
        in_lh = in_lh + pickle.load(f)

    gt_ood = gt_ood + [False] * len(in_lh)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.out_dist}_{args.mode}.pkl', 'rb') as f:
        in_lh = in_lh + pickle.load(f)

    gt_ood = gt_ood + [True] * (len(in_lh) - len(gt_ood))

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.in_dist}_{args.mode}.pkl', 'rb') as f:
        out_lh = out_lh + pickle.load(f)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.out_dist}_{args.mode}.pkl', 'rb') as f:
        out_lh = out_lh + pickle.load(f)

    print(f"in_model: {args.in_model}, mode: {args.mode}, in: {args.in_dist}, out: {args.out_dist}")
    # print("lh:")
    # print(calc_metrics([-m for m in in_lh], gt_ood))
    # print("lr:")
    print(calc_metrics([l - m for m, l in zip(in_lh, out_lh)], gt_ood))

# a given q
else:
    in_lh_q  = []
    out_lh_q     = []
    in_lh_qa = []
    out_lh_qa    = []
    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.in_dist}_q.pkl', 'rb') as f:
        in_lh_q = in_lh_q + pickle.load(f)

    gt_ood = gt_ood + [False] * len(in_lh_q)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.out_dist}_q.pkl', 'rb') as f:
        in_lh_q = in_lh_q + pickle.load(f)

    gt_ood = gt_ood + [True] * (len(in_lh_q) - len(gt_ood))

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.in_dist}_q.pkl', 'rb') as f:
        out_lh_q = out_lh_q + pickle.load(f)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.out_dist}_q.pkl', 'rb') as f:
        out_lh_q = out_lh_q + pickle.load(f)


    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.in_dist}_qa.pkl', 'rb') as f:
        in_lh_qa = in_lh_qa + pickle.load(f)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/in_{args.out_dist}_qa.pkl', 'rb') as f:
        in_lh_qa = in_lh_qa + pickle.load(f)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.in_dist}_qa.pkl', 'rb') as f:
        out_lh_qa = out_lh_qa + pickle.load(f)

    with open(f'./likelihoods/{os.path.basename(args.in_model)}/out_{args.out_dist}_qa.pkl', 'rb') as f:
        out_lh_qa = out_lh_qa + pickle.load(f)

    print(f"in_model: {args.in_model}, mode: {args.mode}, in: {args.in_dist}, out: {args.out_dist}")
    # print("lh:")
    # print(calc_metrics([-mqa + mq for mq, mqa in zip(in_lh_q, in_lh_qa)], gt_ood))
    # print("lr:")
    print(calc_metrics([lqa - mqa + mq - lq for mq, mqa, lq, lqa in zip(in_lh_q, in_lh_qa, out_lh_q, out_lh_qa)], gt_ood))

