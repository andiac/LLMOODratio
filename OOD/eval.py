r"""
Eval metrics for OOD detection.
  Example to run this script:
    python eval.py \
     --dump_path ROSTD_roberta-large.pkl
"""

import os
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser
import pdb
import joblib
#from sklearn.metrics import roc_auc_score
from ood_metrics import calc_metrics
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')


def get_args():
    parser = ArgumentParser(description="OOD metrics eval")
    parser.add_argument(
        "--dump_path",
        type=str,
        default=None,
        required=True,
        help="Path to store (val_likelihoods, val_labels, test_likelihoods, test_labels)",
    )
    parser.add_argument(
        "--evaluate_val",
        default=False,
        action='store_true',
        help="wether to evaluate val datatset metrics",
    )
    parser.add_argument(
        "--use_fp16",
        default=False,
        action='store_true',
        help="wether to use fp16 for inference",
    )

    arguments = parser.parse_args()
    return arguments


def main(args):
    (val_likelihoods_all, val_labels, test_likelihoods_all, test_labels) = joblib.load(args.dump_path)
    val_likelihoods_all = np.array(val_likelihoods_all)
    test_likelihoods_all = np.array(test_likelihoods_all)
    # replace nan with 1e9
    try:
        val_likelihoods_all[np.isnan(val_likelihoods_all)] = -1e9
    except:
        pass
    try:
        test_likelihoods_all[np.isnan(test_likelihoods_all)] = -1e9
    except:
        pass
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # pretrained and finetuned avg
    for i, dtype in enumerate(["fp16"]):
        if i >= len(test_likelihoods_all):
            break
        if args.evaluate_val and val_likelihoods_all:
            cur_val_likelihoods = val_likelihoods_all[i]
            val_likelihoods = cur_val_likelihoods[:, 0] - cur_val_likelihoods[:, 1]
            val_metrics = calc_metrics(val_likelihoods, val_labels)
            # print(f'pretrained and finetuned avg val_metrics({dtype}):')
            print(f'NLR:')
            pprint(val_metrics)

        cur_test_likelihoods = test_likelihoods_all[i]
        test_likelihoods = cur_test_likelihoods[:, 0] - cur_test_likelihoods[:, 1]
        test_metrics = calc_metrics(test_likelihoods, test_labels)
        print(f'NLR:')
        pprint(test_metrics)

        if i == 0:
            print('*'*20)

    # pretrained and finetuned sum
    print('-'*20)
    # for i, dtype in enumerate(["fp32", "fp16"]):
    for i, dtype in enumerate(["."]):
        if i >= len(test_likelihoods_all):
            break
        if args.evaluate_val and val_likelihoods_all:
            cur_val_likelihoods = val_likelihoods_all[i]
            val_likelihoods = cur_val_likelihoods[:, 2] - cur_val_likelihoods[:, 3]
            val_metrics = calc_metrics(val_likelihoods, val_labels)
            print(f'LR:')
            pprint(val_metrics)

        cur_test_likelihoods = test_likelihoods_all[i]
        test_likelihoods = cur_test_likelihoods[:, 2] - cur_test_likelihoods[:, 3]
        test_metrics = calc_metrics(test_likelihoods, test_labels)
        print(f'LR:')
        pprint(test_metrics)

        if i == 0:
            print('*'*20)

    # finetuned avg
    print('-'*20)
    # for i, dtype in enumerate(["fp32", "fp16"]):
    for i, dtype in enumerate(["."]):
        if i >= len(test_likelihoods_all):
            break
        if args.evaluate_val and val_likelihoods_all:
            cur_val_likelihoods = val_likelihoods_all[i]
            val_likelihoods = -cur_val_likelihoods[:, 1]
            val_metrics = calc_metrics(val_likelihoods, val_labels)
            print(f'NLH:')
            pprint(val_metrics)

        cur_test_likelihoods = test_likelihoods_all[i]
        test_likelihoods = -cur_test_likelihoods[:, 1]
        test_metrics = calc_metrics(test_likelihoods, test_labels)
        print(f'NLH:')
        pprint(test_metrics)

        if i == 0:
            print('*'*20)

    # finetuned sum
    print('-'*20)
    # for i, dtype in enumerate(["fp32", "fp16"]):
    for i, dtype in enumerate(["."]):
        if i >= len(test_likelihoods_all):
            break
        if args.evaluate_val and val_likelihoods_all:
            cur_val_likelihoods = val_likelihoods_all[i]
            val_likelihoods =- cur_val_likelihoods[:, 3]
            val_metrics = calc_metrics(val_likelihoods, val_labels)
            print(f'LH:')
            pprint(val_metrics)

        cur_test_likelihoods = test_likelihoods_all[i]
        test_likelihoods = -cur_test_likelihoods[:, 3]
        test_metrics = calc_metrics(test_likelihoods, test_labels)
        print(f'LH:')
        pprint(test_metrics)

        if i == 0:
            print('*'*20)


if __name__ == "__main__":
    args = get_args()
    main(args)

