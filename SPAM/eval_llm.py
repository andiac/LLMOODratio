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
from sklearn.metrics import roc_auc_score
import warnings
import ood_metrics 
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
        "--indist",
        type=str,
        default=None,
        required=False,
        help="indist: ham or spam",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="lh",
        required=False,
        help="metric: lh or lr",
    )
    arguments = parser.parse_args()
    return arguments


def filter_nan(nlls_diff, labels, name):
    nan_indices = np.argwhere(np.isnan(nlls_diff))
    num_nan = len(nan_indices)
    print(f"Number of NaN in {name}: {num_nan}")
    nlls_diff = np.delete(nlls_diff, nan_indices)
    labels = np.delete(labels, nan_indices)
    return nlls_diff, labels

def main(args):

    (val_likelihoods, val_labels, test_likelihoods, test_labels) = joblib.load(args.dump_path)

    val_likelihoods = np.array(val_likelihoods)
    test_likelihoods = np.array(test_likelihoods)

    # finetuned log likelihood - pretrained log likelihood
    if args.metric == "lr":
        val_likelihoods_diff = val_likelihoods[:, 1] - val_likelihoods[:, 0]
        test_likelihoods_diff = test_likelihoods[:, 1] - test_likelihoods[:, 0]
    else:
        val_likelihoods_diff = val_likelihoods[:, 1]
        test_likelihoods_diff = test_likelihoods[:, 1]

    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    if args.indist == "ham":
        val_likelihoods_diff = -val_likelihoods_diff
        test_likelihoods_diff = -test_likelihoods_diff

    # Remove NaNs    
    val_likelihoods_diff, val_labels = filter_nan(val_likelihoods_diff, val_labels, "val_likelihoods_diff")
    test_likelihoods_diff, test_labels = filter_nan(test_likelihoods_diff, test_labels, "test_likelihoods_diff")

    val_auroc = roc_auc_score(val_labels, val_likelihoods_diff)
    val_auroc2 = ood_metrics.auroc(val_likelihoods_diff, val_labels)
    test_auroc = roc_auc_score(test_labels, test_likelihoods_diff)
    test_auroc2 = ood_metrics.auroc(test_likelihoods_diff, test_labels)


    val_aupr = ood_metrics.aupr(val_likelihoods_diff, val_labels)
    test_aupr = ood_metrics.aupr(test_likelihoods_diff, test_labels)
    val_fpr95 = ood_metrics.fpr_at_95_tpr(val_likelihoods_diff, val_labels)
    test_fpr95 = ood_metrics.fpr_at_95_tpr(test_likelihoods_diff, test_labels)

    print(f'val_auroc:{val_auroc} test_auroc:{test_auroc}')
    print(f'val_auroc2:{val_auroc2} test_auroc:{test_auroc2}')
    print(f'val_aupr:{val_aupr} test_aupr:{test_aupr}')
    print(f'val_fpr95:{val_fpr95} test_fpr95:{test_fpr95}')


if __name__ == "__main__":
    args = get_args()
    main(args)


